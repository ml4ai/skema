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
import optuna
from optuna.trial import TrialState
from utils.bleu_score import calculate_bleu_score

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="configuration file for paths and hyperparameters",
    default="configs/resnet_xfmer_latex_config.json",
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
        LEN_DIM = 128

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


def objective(
    trial, train_dataloader, test_dataloader, val_dataloader, vocab, rank=None
):

    # parameters
    optimizer_type = trial.suggest_categorical("optimizer_type", ["Adam"])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    # gamma = trial.suggest_loguniform("gamma", 0.1, 0.9)
    batch_size = trial.suggest_int("batch_size", low=32, high=128, step=8)
    DROPOUT = trial.suggest_float("DROPOUT", low=0.1, high=0.5, step=0.1)
    EMB_DIM = trial.suggest_int("EMB_DIM", low=64, high=512, step=64)
    ENC_DIM = trial.suggest_int("ENC_DIM", low=64, high=512, step=64)
    DEC_HID_DIM = trial.suggest_int("DEC_HID_DIM", low=64, high=512, step=256)
    if optimizer_type == "Adam":
        beta_1 = 0.7#trial.suggest_float("beta1", low=0.5, high=0.9, step=0.1)
        beta_2 = 0.9#trial.suggest_float("beta2", low=0.5, high=0.999, step=0.1)

    # transformers params
    DIM_FEEDFWD = trial.suggest_int("dim_ff_xfmer", low=64, high=1024, step=64)#config["dim_feedforward_for_xfmer"]
    N_HEADS = trial.suggest_int("n_heads", low=2, high=8, step=2)#config["n_xfmer_heads"]
    N_XFMER_ENCODER_LAYERS = 6#trial.suggest_int("n_enc_layer", low=1, high=8, step=1)#config["n_xfmer_encoder_layers"]
    N_XFMER_DECODER_LAYERS = trial.suggest_int("n_dec_layer", low=2, high=8, step=1)#config["n_xfmer_decoder_layers"]

    EPOCHS = config["epochs"]
    # batch_size = config["batch_size"]
    momentum = config["momentum"]
    # scheduler_type = config["scheduler_type"]
    isScheduler = config["isScheduler"]
    whichScheduler = config["whichScheduler"]
    # step_scheduler = config["step_scheduler"]
    # exponential_scheduler = config["exponential_scheduler"]
    # reduce_on_plateau_scheduler = config["ReduceLROnPlateau"]
    starting_lr = config["starting_lr"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    CLIP = config["clip"]
    SEED = config["seed"]
    model_type = config["model_type"]
    g2p = config["garbage2pad"]
    use_single_gpu = config["use_single_gpu"]
    ddp = config["DDP"]
    dataparallel = config["DataParallel"]
    world_size = config["world_size"]

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

            # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
            device = torch.device(
                f"cuda:{config['gpu_id']}"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = define_model(config, vocab, device).to(device)

        elif dataparallel:
            # os.environ["CUDA_VISIBLE_DEVICES"] = dataParallel_ids
            device = torch.device(
                f"cuda:{config['gpu_id']}"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = define_model(config, vocab, device)
            model = nn.DataParallel(
                model.cuda(),
                device_ids=[
                    int(i) for i in config["DataParallel_ids"].split(",")
                ],
            )

        elif ddp:
            # create default process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            # add rank to config
            config["rank"] = rank
            device = f"cuda:{rank}"

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
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # optimizer
    isScheduler = False
    scheduler = None
    # if step_scheduler or exponential_scheduler:
    #     _lr = starting_lr
    #     isScheduler = True
    # else:
    _lr = learning_rate

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
    # which scheduler, if using
    # if step_scheduler:
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=step_size,
    #         gamma=gamma,
    #     )
    # elif exponential_scheduler:
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer,
    #         gamma=gamma,
    #         last_epoch=-1,
    #         verbose=False,
    #     )

    best_valid_loss = float("inf")
    trg_pad_idx = vocab.stoi["<pad>"]

    # raw data paths
    img_tnsr_path = (
        f"{config['data_path']}/{config['dataset_type']}/image_tensors"
    )

    for epoch in range(EPOCHS):

        start_time = time.time()
        print("epoch: ", epoch)
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
                    isScheduler=isScheduler,
                    scheduler=scheduler,
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
            is_test=True,
            g2p=g2p,
        )


        # calculate bleu score
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    bs = calculate_bleu_score()

    return bs


def tune(rank=None,):

    config["rank"] = rank
    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        vocab,
    ) = preprocess_dataset(config)

    func = lambda trial: objective(
        trial, train_dataloader, test_dataloader, val_dataloader, vocab, rank
    )

    if config["DDP"] and rank==0:
        study = optuna.create_study(direction="maximize")
        study.optimize(func, n_trials=20)

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ == "__main__":
    if config["DDP"]:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29880"
        world_size = config["world_size"]
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["DDP gpus"]
        mp.spawn(tune, args=(), nprocs=world_size, join=True)
    else:
        tune()
