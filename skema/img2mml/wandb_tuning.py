# -*- coding: utf-8 -*-
"Main script to train the model."

import os
import random
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

import wandb

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="configuration file for paths and hyperparameters",
    default="configs/resnet_xfmer_mml_config.json",
)

args = parser.parse_args()

with open(args.config, "r") as cfg:
    main_config = json.load(cfg)

torch.backends.cudnn.enabled = False


def set_random_seed(SEED):
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def define_model(main_config,sweep_config, VOCAB, DEVICE):
    """
    defining the model
    initializing encoder, decoder, and model
    """

    print("defining model...")

    MODEL_TYPE = main_config["model_type"]
    INPUT_CHANNELS = main_config["input_channels"]
    OUTPUT_DIM = len(VOCAB)
    EMB_DIM = sweep_config["embedding_dim"]
    ENC_DIM = sweep_config["encoder_dim"]
    DEC_HID_DIM = sweep_config["decoder_hid_dim"]
    DROPOUT = sweep_config["dropout"]
    MAX_LEN = main_config["max_len"]

    print(f"building {MODEL_TYPE} model...")

    if MODEL_TYPE == "opennmt":
        ENCODING_TYPE = "row_encoding"
        N_LAYERS = main_config["lstm_layers"]
        TFR = main_config["teacher_force_ratio"]
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
        DIM_FEEDFWD = sweep_config["dim_feedforward_for_xfmer"]
        N_HEADS = sweep_config["n_xfmer_heads"]
        N_XFMER_ENCODER_LAYERS = sweep_config["n_xfmer_encoder_layers"]
        N_XFMER_DECODER_LAYERS = sweep_config["n_xfmer_decoder_layers"]
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
        DIM_FEEDFWD = sweep_config["dim_feedforward_for_xfmer"]
        N_HEADS = sweep_config["n_xfmer_heads"]
        N_XFMER_ENCODER_LAYERS = sweep_config["n_xfmer_encoder_layers"]
        N_XFMER_DECODER_LAYERS = sweep_config["n_xfmer_decoder_layers"]
        LEN_DIM = 128  # 32

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


def train_model(rank=None, config=None,):

    # to save trained model and logs
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

    with wandb.init(config=config, group="DDP"):
        sweep_config = wandb.config

        # parameters
        EPOCHS = main_config["epochs"]

        batch_size = sweep_config["batch_size"]
        main_config["batch_size"] = batch_size

        optimizer_type = main_config["optimizer_type"]
        learning_rate = sweep_config["learning_rate"]
        weight_decay = main_config["weight_decay"]
        scheduler_type = main_config["scheduler_type"]
        step_scheduler = main_config["step_scheduler"]
        exponential_scheduler = main_config["exponential_scheduler"]
        reduce_on_plateau_scheduler = main_config["ReduceLROnPlateau"]
        starting_lr = main_config["starting_lr"]
        step_size = main_config["step_size"]
        gamma = main_config["gamma"]
        (beta_1, beta_2) = sweep_config["beta_1"], sweep_config["beta_2"]
        momentum = main_config["momentum"]
        CLIP = main_config["clip"]
        SEED = main_config["seed"]
        min_length_bean_search_normalization = main_config[
            "min_length_bean_search_normalization"
        ]
        alpha = main_config["beam_search_alpha"]
        beam_k = main_config["beam_k"]
        model_type = main_config["model_type"]
        dataset_type = main_config["dataset_type"]
        load_trained_model_for_testing = main_config["testing"]
        cont_training = main_config["continue_training_from_last_saved_model"]
        g2p = main_config["garbage2pad"]
        use_single_gpu = main_config["use_single_gpu"]
        ddp = main_config["DDP"]
        dataparallel = main_config["DataParallel"]
        dataParallel_ids = main_config["DataParallel_ids"]
        world_size = main_config["world_size"]
        early_stopping = main_config["early_stopping"]
        early_stopping_counts = main_config["early_stopping_counts"]


        # set_random_seed
        set_random_seed(SEED)

        # defining model using DataParallel
        if torch.cuda.is_available() and main_config["device"] == "cuda":
            if use_single_gpu:
                print(f"using single gpu:{main_config['gpu_id']}...")

                os.environ["CUDA_VISIBLE_DEVICES"] = str(main_config["gpu_id"])
                device = torch.device(
                    f"cuda:{main_config['gpu_id']}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                (
                    train_dataloader,
                    test_dataloader,
                    val_dataloader,
                    vocab,
                ) = preprocess_dataset(main_config)
                model = define_model(main_config, sweep_config, vocab, device).to(device)

            elif dataparallel:
                os.environ["CUDA_VISIBLE_DEVICES"] = dataParallel_ids
                device = torch.device(
                    f"cuda:{main_config['gpu_id']}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                (
                    train_dataloader,
                    test_dataloader,
                    val_dataloader,
                    vocab,
                ) = preprocess_dataset(main_config)
                model = define_model(main_config,sweep_config, vocab, device)
                model = nn.DataParallel(
                    model.cuda(),
                    device_ids=[
                        int(i) for i in main_config["DataParallel_ids"].split(",")
                    ],
                )

            elif ddp:
                # create default process group
                dist.init_process_group("nccl", rank=rank, world_size=world_size)
                # add rank to main_config
                main_config["rank"] = rank
                device = f"cuda:{rank}"
                (
                    train_dataloader,
                    test_dataloader,
                    val_dataloader,
                    vocab,
                ) = preprocess_dataset(main_config)
                model = define_model(main_config,sweep_config, vocab, rank)
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
            ) = preprocess_dataset(main_config,batch_size)
            model = define_model(main_config,sweep_config, vocab, device).to(device)


        print("MODEL: ")
        print(f"The model has {count_parameters(model)} trainable parameters")

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

        # intializing loss function
        criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

        best_valid_loss = float("inf")

        # raw data paths
        img_tnsr_path = (
            f"{main_config['data_path']}/{main_config['dataset_type']}/image_tensors"
        )

        if not load_trained_model_for_testing:
            count_es = 0
            # if continue_training_from_last_saved_model
            # model will be lastest saved model
            if cont_training:
                model.load_state_dict(
                    torch.load(
                        f"trained_models/{model_type}_{dataset_type}_{main_config['markup']}_latest.pt"
                    )
                )
                print("continuing training from lastest saved model...")

            for epoch in range(EPOCHS):
                if count_es <= early_stopping_counts:
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
                    if (not ddp) or (ddp and rank == 0):
                        wandb.log({"val loss": val_loss, "epoch": epoch})

        if ddp:
            dist.destroy_process_group()

        time.sleep(3)

"============================================================"
def ddp_main(config=None):

    print("in dpp: ", config)
    world_size = main_config["world_size"]
    os.environ["CUDA_VISIBLE_DEVICES"] = main_config["DDP gpus"]
    mp.spawn(train_model, args=(config,), nprocs=world_size, join=True)

if __name__ == "__main__":

    wandb.login()

    # Define the search space
    sweep_config = {"method":"random"}
    metric = {
        "name":"loss",
        "goal":"minimize"
    }

    sweep_config["metric"] = metric

    parameters_dict = {
        'encoder_dim': {
            'values': [128, 256, 512]
            },
        'embedding_dim': {
            'values': [128, 256, 512]
            },
        'decoder_hid_dim': {
            'values': [128, 256, 512]
            },
        'dim_feedforward_for_xfmer': {
            'values': [128, 256, 512]
            },
        'dropout': {
              'values': [0.1,0.2,0.3]
            },
        'n_xfmer_heads':{
                'values': [2,4,6]
            },
        'n_xfmer_encoder_layers':{
            'values':[2,3,4,5,6]
        },
        'n_xfmer_decoder_layers':{
            'values':[2,3,4,5,6]
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
         },
         'beta_1': {
             # a flat distribution between 0 and 0.1
             'distribution': 'uniform',
             'min': 0.5,
             'max': 0.9
          },
          'beta_2': {
              # a flat distribution between 0 and 0.1
              'distribution': 'uniform',
              'min': 0.5,
              'max': 0.999
           },
          'batch_size': {
              'values': [128]
          },
        # 'batch_size': {
        #     # a flat distribution between 0 and 0.1
        #     'distribution': 'q_log_uniform_values',
        #     'q': 8,
        #     'min': 32,
        #     'max': 128,
        #  },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    train_model(sweep_config)

    wandb.agent(sweep_id, train_model, count=1)
    # wandb.agent(sweep_id, ddp_main, count=1)
