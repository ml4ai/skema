# -*- coding: utf-8 -*-
"Main script to train the model."

import os, random
import numpy as np
import time
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from skema.img2mml.preprocessing.preprocess_dataset import preprocess_dataset
from skema.img2mml.models.encoders.cnn_encoder import CNN_Encoder
from skema.img2mml.models.encoders.resnet_encoder import (
    ResNet18_Encoder,
    ResNetBlock,
)
from skema.img2mml.models.encoders.xfmer_encoder import Transformer_Encoder
from skema.img2mml.models.decoders.lstm_decoder import LSTM_Decoder
from skema.img2mml.models.decoders.xfmer_decoder import Transformer_Decoder
from skema.img2mml.models.image2mml_lstm import Image2MathML_LSTM
from skema.img2mml.models.image2mml_xfmer import Image2MathML_Xfmer
from skema.img2mml.src.train import train
from skema.img2mml.src.test import evaluate
import re

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["arxiv", "im2mml", "arxiv_im2mml"],
    default="arxiv_im2mml",
    help="Choose which dataset to be used for training. Choices: arxiv, im2mml, arxiv_im2mml.",
)
parser.add_argument(
    "--with_fonts",
    action="store_true",
    default=False,
    help="Whether using the dataset with diverse fonts",
)
parser.add_argument(
    "--with_boldface",
    action="store_true",
    default=False,
    help="Whether having boldface in labels",
)
parser.add_argument(
    "--config",
    help="configuration file for paths and hyperparameters",
    default="configs/xfmer_mml_config.json",
)

args = parser.parse_args()

dataset = args.dataset
if args.with_fonts:
    dataset += "_with_fonts"

with open(args.config, "r") as cfg:
    config = json.load(cfg)
    config["dataset"] = dataset
    if args.with_boldface:
        config["with_boldface"] = "True"
    else:
        config["with_boldface"] = "False"

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
        model = Image2MathML_LSTM(ENC, DEC, DEVICE, ENCODING_TYPE, MAX_LEN, VOCAB)

    elif MODEL_TYPE == "cnn_xfmer":
        # transformers params
        # ENCODING_TYPE will be PositionalEncoding
        DIM_FEEDFWD = config["dim_feedforward_for_xfmer"]
        N_HEADS = config["n_xfmer_heads"]
        N_XFMER_ENCODER_LAYERS = config["n_xfmer_encoder_layers"]
        N_XFMER_DECODER_LAYERS = config["n_xfmer_decoder_layers"]
        LEN_DIM = 930

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
                LEN_DIM,
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
        LEN_DIM = 32

        ENC = {
            "CNN": ResNet18_Encoder(
                INPUT_CHANNELS, DEC_HID_DIM, DROPOUT, DEVICE, ResNetBlock
            ),
            "XFMER": Transformer_Encoder(
                EMB_DIM,
                DEC_HID_DIM,
                N_HEADS,
                DROPOUT,
                DEVICE,
                MAX_LEN,
                N_XFMER_ENCODER_LAYERS,
                DIM_FEEDFWD,
                LEN_DIM,
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
    initializing the model weights with values
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


def get_last_epoch(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    pattern = r"Epoch: (\d+) \| Time:"
    matches = re.findall(pattern, content)

    if matches:
        last_epoch = int(matches[-1])
        return last_epoch

    return 0


def train_model(
    rank=None,
):
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
    load_trained_model_for_testing = config["load_trained_model_for_testing"]
    cont_training = config["continue_training_from_last_saved_model"]
    g2p = config["garbage2pad"]
    use_single_gpu = config["use_single_gpu"]
    ddp = config["DDP"]
    dataparallel = config["DataParallel"]
    dataParallel_ids = config["DataParallel_gpu_ids"]
    num_DDP_gpus = config["num_DDP_gpus"]
    early_stopping = config["early_stopping"]
    early_stopping_counts = config["early_stopping_counts"]
    minimum_training_epochs = config["minimum_training_epochs"]

    # set_random_seed
    set_random_seed(SEED)

    # to save trained model and logs
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

    current_epoch = 0
    if args.with_boldface:
        # to log losses
        if cont_training:
            loss_file = open(f"logs/{dataset}_boldface_loss_file.txt", "a")
            current_epoch = get_last_epoch(f"logs/{dataset}_boldface_loss_file.txt")
        else:
            loss_file = open(f"logs/{dataset}_boldface_loss_file.txt", "w")
        # to log config(to keep track while running multiple experiments)
        config_log = open(f"logs/{dataset}_boldface_config_log.txt", "w")
    else:
        # to log losses
        if cont_training:
            loss_file = open(f"logs/{dataset}_loss_file.txt", "a")
            current_epoch = get_last_epoch(f"logs/{dataset}_loss_file.txt")
        else:
            loss_file = open(f"logs/{dataset}_loss_file.txt", "w")
        # to log config(to keep track while running multiple experiments)
        config_log = open(f"logs/{dataset}_config_log.txt", "w")

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
            dist.init_process_group("nccl", rank=rank, world_size=num_DDP_gpus)
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
    print(f"The model has {count_parameters(model):,} trainable parameters")

    # initializing loss function
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
    trg_pad_idx = vocab.stoi["<pad>"]

    # raw data paths
    img_tnsr_path = f"{config['data_path']}/{config['dataset_type']}/{config['dataset']}/image_tensors"

    if not load_trained_model_for_testing:
        count_es = 0
        # if continue_training_from_last_saved_model
        # model will be latest saved model
        if cont_training:
            if args.with_boldface:
                model.load_state_dict(
                    torch.load(
                        f"trained_models/{model_type}_{dataset}_boldface_latest.pt"
                    )
                )
            else:
                model.load_state_dict(
                    torch.load(f"trained_models/{model_type}_{dataset}_latest.pt")
                )

            print("continuing training from latest saved model...")

        for epoch in range(current_epoch, EPOCHS):
            if count_es <= early_stopping_counts or epoch < minimum_training_epochs:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    model_type,
                    img_tnsr_path,
                    train_dataloader,
                    optimizer,
                    criterion,
                    CLIP,
                    device,
                    ddp=ddp,
                    rank=rank,
                    vocab=vocab
                )

                val_loss = evaluate(
                    model,
                    model_type,
                    img_tnsr_path,
                    batch_size,
                    val_dataloader,
                    criterion,
                    device,
                    vocab,
                    ddp=ddp,
                    rank=rank,
                    g2p=g2p,
                )

                end_time = time.time()
                # total time spent on training an epoch
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                # saving the current model for transfer learning
                if (not ddp) or (ddp and rank == 0):
                    if args.with_boldface:
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset}_boldface_latest.pt",
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset}_latest.pt",
                        )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    if args.with_boldface:
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset}_boldface_best.pt",
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset}_best.pt",
                        )
                    count_es = 0

                elif early_stopping:
                    count_es += 1

                # logging
                if (not ddp) or (ddp and rank == 0):
                    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
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

            else:
                print(
                    f"Terminating the training process as the validation loss hasn't been reduced from last {early_stopping_counts} epochs."
                )
                break

        if args.with_boldface:
            print(
                "best model saved as:  ",
                f"trained_models/{model_type}_{dataset}_boldface_best.pt",
            )
        else:
            print(
                "best model saved as:  ",
                f"trained_models/{model_type}_{dataset}_best.pt",
            )

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    if args.with_boldface:
        print(
            "loading best saved model: ",
            f"trained_models/{model_type}_{dataset}_boldface_best.pt",
        )
    else:
        print(
            "loading best saved model: ",
            f"trained_models/{model_type}_{dataset}_best.pt",
        )
    try:
        # loading pre_trained_model
        if args.with_boldface:
            model.load_state_dict(
                torch.load(f"trained_models/{model_type}_{dataset}_boldface_best.pt")
            )
        else:
            model.load_state_dict(
                torch.load(f"trained_models/{model_type}_{dataset}_best.pt")
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
                f"module.{key}": value for key, value in model.state_dict().items()
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
        img_tnsr_path,
        batch_size,
        test_dataloader,
        criterion,
        device,
        vocab,
        beam_params=beam_params,
        is_test=True,
        ddp=ddp,
        rank=rank,
        g2p=g2p,
        config=config
    )

    if (not ddp) or (ddp and rank == 0):
        print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
        )

    # stopping time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


# for DDP
def ddp_main():
    num_DDP_gpus = config["num_DDP_gpus"]
    os.environ["CUDA_VISIBLE_DEVICES"] = config["DDP_gpu_ids"]
    mp.spawn(train_model, args=(), nprocs=num_DDP_gpus, join=True)


if __name__ == "__main__":
    if config["DDP"]:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        ddp_main()
    else:
        train_model()
