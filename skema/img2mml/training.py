# -*- coding: utf-8 -*-
"Main script to train the model."

import os, random
import subprocess
import numpy as np
import time
import json
import math
import argparse
import logging
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from skema.img2mml.preprocessing.preprocess import preprocess_dataset
from skema.img2mml.models.encoders.cnn_encoder import CNN_Encoder
from skema.img2mml.models.encoders.resnet_encoder import ResNet18_Encoder, ResNetBlock
from skema.img2mml.models.encoders.xfmer_encoder import Transformer_Encoder
from skema.img2mml.models.decoders.lstm_decoder import LSTM_Decoder
from skema.img2mml.models.decoders.xfmer_decoder import Transformer_Decoder
from skema.img2mml.models.image2mml_lstm import Image2MathML_LSTM
from skema.img2mml.models.image2mml_xfmer import Image2MathML_Xfmer
from skema.img2mml.src.train import train
from skema.img2mml.src.test import evaluate

# from utils.bleu_score import calculate_bleu_score

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="configuration file for paths and hyperparameters",
    default="configs/xfmer_mml_config.json",
)

args = parser.parse_args()

with open(args.config, "r") as cfg:
    config = json.load(cfg)

torch.backends.cudnn.enabled = False

def set_random_seed(SEED):
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def define_model(config, VOCAB, DEVICE):
    """
    defining the model
    initializing encoder, decoder, and model
    """

    print("defining model...")

    MODEL_TYPE = config["model_type"]
    INPUT_CHANNELS = config["input_channels"]
    OUTPUT_DIM = len(VOCAB)
    EMB_DIM = config["embedding_dim"]
    ENC_DIM = config["encoder_dim"]
    # ENCODING_TYPE = config["encoding_type"]
    DEC_HID_DIM = config["decoder_hid_dim"]
    DROPOUT = config["dropout"]
    MAX_LEN = config["max_len"]

    print(f"building {MODEL_TYPE} model...")

    if MODEL_TYPE == "opennmt":
        ENCODING_TYPE = "row_encoding"
        N_LAYERS = config["lstm_layers"]
        TFR = config["teacher_force_ratio"]
        ENC = CNN_Encoder(INPUT_CHANNELS, DEC_HID_DIM, DROPOUT, DEVICE)
        DEC = LSTM_Decoder(
            EMB_DIM,
            ENC_DIM,
            DEC_HID_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            DEVICE,
            DROPOUT,
            TFR,
        )
        model = Image2MathML_LSTM(
            ENC, DEC, DEVICE, ENCODING_TYPE, MAX_LEN, VOCAB
        )

    elif MODEL_TYPE == "cnn_xfmer":
        # transformers params
        # ENCODING_TYPE will be PositionalEncoding
        DIM_FEEDFWD = config["dim_feedforward_for_xfmer"]
        N_HEADS = config["n_xfmer_heads"]
        N_XFMER_ENCODER_LAYERS = config["n_xfmer_encoder_layers"]
        N_XFMER_DECODER_LAYERS = config["n_xfmer_decoder_layers"]
        LEN_DIM=930

        ENC = {
            "CNN": CNN_Encoder(INPUT_CHANNELS, DEC_HID_DIM, DROPOUT, DEVICE),
            "XFMER": Transformer_Encoder(
                EMB_DIM,
                DEC_HID_DIM,
                N_HEADS,
                DROPOUT,
                DEVICE,
                MAX_LEN,
                N_XFMER_ENCODER_LAYERS,
                DIM_FEEDFWD,
                LEN_DIM
            ),
        }
        DEC = Transformer_Decoder(
            EMB_DIM,
            N_HEADS,
            DEC_HID_DIM,
            OUTPUT_DIM,
            DROPOUT,
            MAX_LEN,
            N_XFMER_DECODER_LAYERS,
            DIM_FEEDFWD,
            DEVICE,
        )

        model = Image2MathML_Xfmer(ENC, DEC, VOCAB, DEVICE)

    elif MODEL_TYPE == "resnet_xfmer":
        # transformers params
        DIM_FEEDFWD = config["dim_feedforward_for_xfmer"]
        N_HEADS = config["n_xfmer_heads"]
        N_XFMER_ENCODER_LAYERS = config["n_xfmer_encoder_layers"]
        N_XFMER_DECODER_LAYERS = config["n_xfmer_decoder_layers"]
        LEN_DIM=32

        ENC = {
            "CNN": ResNet18_Encoder(INPUT_CHANNELS, DEC_HID_DIM, DROPOUT, DEVICE, ResNetBlock),
            "XFMER": Transformer_Encoder(
                EMB_DIM,
                DEC_HID_DIM,
                N_HEADS,
                DROPOUT,
                DEVICE,
                MAX_LEN,
                N_XFMER_ENCODER_LAYERS,
                DIM_FEEDFWD,
                LEN_DIM
            ),
        }
        DEC = Transformer_Decoder(
            EMB_DIM,
            N_HEADS,
            DEC_HID_DIM,
            OUTPUT_DIM,
            DROPOUT,
            MAX_LEN,
            N_XFMER_DECODER_LAYERS,
            DIM_FEEDFWD,
            DEVICE,
        )

        model = Image2MathML_Xfmer(ENC, DEC, VOCAB, DEVICE)

    return model


def init_weights(m):
    """
    initializing the model wghts with values
    drawn from normal distribution.
    else initialize them with 0.
    """
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """
    counting total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    """
    epoch timing
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(rank=None,):

    # parameters
    EPOCHS = config["epochs"]
    batch_size = config["batch_size"]
    optimizer_type = config["optimizer_type"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    use_scheduler = config["use_scheduler"]
    starting_lr = config["starting_lr"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    (beta_1, beta_2) = config["beta_1"], config["beta_2"]
    momentum = config["momentum"]
    CLIP = config["clip"]
    SEED = config["seed"]
    min_length_bean_search_normalization = config[
        "min_length_bean_search_normalization"
    ]
    alpha = config["beam_search_alpha"]
    beam_k = config["beam_k"]
    model_type = config["model_type"]
    dataset_type = config["dataset_type"]
    load_trained_model = config["load_trained_model_for_testing"]
    cont_training = config["continue_training_from_last_saved_model"]
    g2p = config["garbage2pad"]
    use_single_gpu = config["use_single_gpu"]
    ddp = config["DDP"]
    dataparallel = config["DataParallel"]
    dataParallel_ids = config["DataParallel_ids"]
    world_size = config["world_size"]
    early_stopping = config["early_stopping"]
    early_stopping_counts = config["early_stopping_counts"]

    # set_random_seed
    set_random_seed(SEED)

    # to save trained model and logs
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

    # to log losses
    loss_file = open("logs/loss_file.txt", "w")
    # to log config(to keep track while running multiple experiments)
    config_log = open("logs/config_log.txt", "w")
    json.dump(config, config_log)

    # defining model using DataParallel
    if torch.cuda.is_available():
        if use_single_gpu:
            print(f"using single gpu:{config['gpu_id']}...")

            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
            device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(config)
            model = define_model(config, vocab, device).to(device)

        elif dataparallel:
            os.environ["CUDA_VISIBLE_DEVICES"] = dataParallel_ids
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(config)
            model = define_model(config, vocab, device)
            model = nn.DataParallel(
                model.cuda(),
                device_ids=[int(i) for i in config["DataParallel_ids"].split(",")],
            )

        elif ddp:
            # create default process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            # add rank to config
            config["rank"] = rank
            device = f"cuda:{rank}"
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(config)
            model = define_model(config, vocab, rank)
            model = DDP(
                model.to(f"cuda:{rank}"),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

    else:
        import warnings

        warnings.warn("No GPU input has provided. Falling back to CPU. ")
        device = torch.device("cpu")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            vocab,
        ) = preprocess_dataset(config)
        model = define_model(config, vocab, device).to(device)

    print("MODEL: ")
    # print(model.apply(init_weights))
    print(f"The model has {count_parameters(model):,} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # optimizer
    _lr = starting_lr if use_scheduler else learning_rate

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=_lr,
            weight_decay=weight_decay,
            betas=(beta_1, beta_2),
        )
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    best_valid_loss = float("inf")
    TRG_PAD_IDX = vocab.stoi["<pad>"]

    if not load_trained_model:
        count_es = 0
        # if continue_training_from_last_saved_model
        # model will be lastest saved model
        if cont_training:
            model.load_state_dict(
                torch.load(
                    f"trained_models/{model_type}_{dataset_type}_latest.pt"
                )
            )
            print("continuing training from lastest saved model...")

        for epoch in range(EPOCHS):
            # if (not ddp) or (ddp and rank == 0):
            #     print(f"Epoch: {epoch + 1} / {EPOCHS}")

            if count_es <= early_stopping_counts:

                start_time = time.time()

                # training and validation
                # with tqdm(train_dataloader, unit="batch", desc="training") as tepoch:
                # train_loss = train(model, model_type, tepoch, vocab,
                #                  optimizer, CLIP, ddp=ddp, rank=rank)
                train_loss = train(
                    model,
                    model_type,
                    train_dataloader,
                    optimizer,
                    criterion,
                    CLIP,
                    device,
                    ddp=ddp,
                    rank=rank,
                )

                # dist.barrier()  # using single gpu
                # if rank==0:
                val_loss = evaluate(
                    model,
                    model_type,    
                    batch_size,
                    val_dataloader,
                    criterion,
                    device,
                    ddp=ddp,
                    rank=rank,
                    g2p=g2p,
                )

                end_time = time.time()
                # total time spent on training an epoch
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                # if (not ddp) or (ddp and rank==0):
                #     torch.save(model.state_dict(), f'trained_models/{model_type}_{dataset_type}_latest.pt')

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    if (not ddp) or (ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset_type}_best.pt",
                        )
                        count_es = 0

                elif early_stopping:
                    count_es += 1

                # logging
                if (not ddp) or (ddp and rank == 0):
                    print(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
                    )
                    print(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
                    )
                    print(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}"
                    )

                    loss_file.write(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    )
                    loss_file.write(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n"
                    )
                    loss_file.write(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n"
                    )

                # dist.barrier()

            else:
                print(
                    f"Terminating the training process as the validation loss hasn't been reduced from last {early_stopping_counts} epochs."
                )
                break

        print(
            "best model saved as:  ",
            f"trained_models/{model_type}_{dataset_type}_best.pt",
        )

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/{model_type}_{dataset_type}_best.pt",
    )
    try:
        # loading pre_tained_model
        model.load_state_dict(
            torch.load(f"trained_models/{model_type}_{dataset_type}_best.pt")
        )
    except:
        try:
            # removing "module." from keys
            pretrained_dict = {
                key.replace("module.", ""): value
                for key, value in model.state_dict().items()
            }
        except:
            # adding "module." in keys
            pretrained_dict = {
                f"module.{key}": value
                for key, value in model.state_dict().items()
            }

        model.load_state_dict(pretrained_dict)

    epoch = "test_0"
    if config["beam_search"]:
        beam_params = [beam_k, alpha, min_length_bean_search_normalization]
    else:
        beam_params = None

    test_loss = evaluate(
        model,
        model_type,
        vocab,
        batch_size,
        test_dataloader,
        criterion,
        device,
        beam_params,
        is_test=True,
        ddp=ddp,
        rank=rank,
        g2p=g2p,
    )

    if (not ddp) or (ddp and rank == 0):
        print(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
        )
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
        )

    # stopping time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

# for DDP
def ddp_main():
    world_size = config["world_size"]
    os.environ["CUDA_VISIBLE_DEVICES"] = config["DDP gpus"]
    mp.spawn(train_model, args=(), nprocs=world_size, join=True)


if __name__ == "__main__":

    if config["DDP"]:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        ddp_main()
    else:
        train_model()
