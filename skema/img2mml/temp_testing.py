import pandas as pd
import random
import torch
import os, sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from functools import partial
import subprocess
import numpy as np
import time
import json
import math
import argparse
import logging
import itertools
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

# opening config file

with open("configs/xfmer_latex_config.json", "r") as cfg:
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


class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        eqn = self.dataframe.iloc[index, 1]
        indexed_eqn = []
        for token in eqn.split():
            if self.vocab.stoi[token] != None:
                indexed_eqn.append(self.vocab.stoi[token])
            else:
                indexed_eqn.append(self.vocab.stoi["<unk>"])

        return self.dataframe.iloc[index, 0], torch.Tensor(indexed_eqn)


class My_pad_collate(object):
    """
    padding mml to max_len, and stacking images
    return: mml_tensors of shape [batch, max_len]
            stacked image_tensors [batch]
    """

    def __init__(self, device, vocab, max_len):
        self.device = device
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.stoi["<pad>"]

    def __call__(self, batch):
        _img, _mml = zip(*batch)

        # padding mml
        # padding to a fix max_len equations with more tokens than
        # max_len will be chopped down to max_length.

        batch_size = len(_mml)
        padded_mml_tensors = (
            torch.ones([batch_size, self.max_len], dtype=torch.long)
            * self.pad_idx
        )
        for b in range(batch_size):
            if len(_mml[b]) <= self.max_len:
                padded_mml_tensors[b][: len(_mml[b])] = _mml[b]
            else:
                padded_mml_tensors[b][: self.max_len] = _mml[b][: self.max_len]

        # images tensors
        _img = torch.Tensor(_img)

        return (
            _img.to(self.device),
            padded_mml_tensors.to(self.device),
        )


def preprocess_dataset(device, max_len, start=None, end=None, ):

    # reading raw text files
    img_tnsr_path = f"{config['data_path']}/{config['dataset_type']}/image_tensors"
    vocab = open("vocab.txt").readlines()

    df = pd.read_csv(f"{config['data_path']}/{config['dataset_type']}/test.csv")
    imgs, eqns = df["IMG"], df["EQUATION"]

    eqns_arr = list()
    imgs_arr = list()
    for i,e in zip(imgs, eqns):
        # if len(e.split()) > start and len(e.split()) <= end:
        eqns_arr.append(e)
        imgs_arr.append(i)

    test = {
        "IMG": imgs_arr,
        "EQUATION": eqns_arr,
    }

    vocab_file = open("vocab.txt").readlines()
    counter = dict()
    for line in vocab_file:
        counter[line.split()[0]] = 1

    vocab = Vocab(counter, min_freq=1)

    # define tokenizer function
    tokenizer = lambda x: x.split()

    # initializing pad collate class
    mypadcollate = My_pad_collate(device, vocab, max_len)

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test, vocab, tokenizer)
    test_dataloader = DataLoader(
        imml_test,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        sampler=None,
        collate_fn=mypadcollate,
        pin_memory=False,
    )

    return test_dataloader, vocab

def test_model():

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
    load_trained_model_for_testing = config["testing"]
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
    device = "cuda:5"
    test_dataloader, vocab = preprocess_dataset(device,config["max_len"])

    model = define_model(config, vocab, device).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # optimizer
    _lr = starting_lr if use_scheduler else learning_rate
    optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=_lr,
            weight_decay=weight_decay,
            betas=(beta_1, beta_2),
        )

    trg_pad_idx = vocab.stoi["<pad>"]

    # raw data paths
    img_tnsr_path = f"{config['data_path']}/{config['dataset_type']}/image_tensors"
    print(
        "loading best saved model: ",
        f"trained_models/{model_type}_{dataset_type}_{config['markup']}_best.pt",
    )
    try:
        # loading pre_tained_model
        model.load_state_dict(
            torch.load(f"trained_models/{model_type}_{dataset_type}_{config['markup']}_best.pt")
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
    )


    print(
        f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
    )

test_model()
