# -*- coding: utf-8 -*-

import os, subprocess
import random
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
from torchvision import transforms
from PIL import Image
from models.encoders.cnn_encoder import CNN_Encoder
from models.decoders.lstm_decoder import LSTM_Decoder
from models.image2mml_lstm import Image2MathML_LSTM
import io
from typing import List
from fastapi import FastAPI, File


class Image2Tensor(object):
    '''
    This class takes in an image and generates a tensor object
    '''
    def __init__(self):
        print("Converting image to torch tensor...")

    def crop_image(self, image, size):
        return transforms.functional.crop(image, 0, 0, size[0], size[1])


    def resize_image(self, image):
        return image.resize((int(image.size[0]/2), int(image.size[1]/2)))

    def pad_image(self, IMAGE):
        right,left,top,bottom = 8, 8, 8, 8
        width, height = IMAGE.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(IMAGE.mode, (new_width, new_height))
        result.paste(IMAGE,(left, top))
        return result


    def __call__(self, image_path):
        """
        convert png image to tensor
        :params: png image path
        :return: processed image tensor
        """

        # crop, resize, and pad the image
        mean_w, mean_h = 500, 50
        IMAGE = Image.open(io.BytesIO(image_path)).convert('L')
        IMAGE = self.crop_image(IMAGE, [mean_h, mean_w])
        IMAGE = self.resize_image(IMAGE)
        IMAGE = self.pad_image(IMAGE)
        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

        return IMAGE


def set_random_seed(SEED: int)-> None:
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def define_model(config: dict, VOCAB: List[str], DEVICE: torch.device) -> Image2MathML_LSTM:
    '''
    defining the model
    initializing encoder, decoder, and model

    '''

    print('defining model...')

    MODEL_TYPE = config["model_type"]
    INPUT_CHANNELS = config["input_channels"]
    OUTPUT_DIM = len(VOCAB)
    EMB_DIM = config["embedding_dim"]
    ENC_DIM = config["encoder_dim"]
    ENCODING_TYPE = config["encoding_type"]
    DEC_HID_DIM = config["decoder_hid_dim"]
    DROPOUT = config["dropout"]
    MAX_LEN = 110

    print(f'building {MODEL_TYPE} model...')

    N_LAYERS = config["lstm_layers"]
    TFR=0
    ENC = CNN_Encoder(INPUT_CHANNELS, DEC_HID_DIM, DROPOUT, DEVICE)
    DEC = LSTM_Decoder(EMB_DIM, ENC_DIM,  DEC_HID_DIM, OUTPUT_DIM, N_LAYERS, DEVICE, DROPOUT, TFR)
    model = Image2MathML_LSTM(ENC, DEC, DEVICE, ENCODING_TYPE, MAX_LEN, VOCAB)

    return model


def evaluate(model: Image2MathML_LSTM, vocab: List[str], img: Image2Tensor, device:torch.device) -> str:
    '''
    It predicts the sequence for the image to translate it into MathML contents
    '''
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = model(img, is_train=False, is_test=False)  # O: (B, max_len, output_dim), preds: (B, max_len)

        # translating mathml
        topK = torch.zeros(output.shape[0], output.shape[1])
        pred_arr = list()
        for j in range(output.shape[1]):
            top = output[:,j,:].argmax(1)
            topK[:,j] = top
            if vocab[top].split()[0] != "<eos>":
                pred_arr.append(vocab[top].split()[0])
            else:
                break
        pred_seq =  " ".join(pred_arr[1:])
        return pred_seq

def count_parameters(model):
    '''
    counting total number of parameters
    '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def render_mml(config: dict, model_path, vocab: List[str], imagetensor) -> str:
    '''
    It allows us to obtain mathML for an image
    '''
    # parameters
    optimizer_type = config["optimizer_type"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    (beta_1, beta_2) = config["beta_1"],config["beta_2"]
    CLIP = config["clip"]
    SEED = config["seed"]
    model_type = config["model_type"]
    load_trained_model = config["load_trained_model_for_testing"]
    use_single_gpu = config["use_single_gpu"]

    # set_random_seed
    set_random_seed(SEED)


    # defining model using DataParallel
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model: Image2MathML_LSTM = define_model(config, vocab, device).to(device)

    # optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params = model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                betas=(beta_1, beta_2))

    best_valid_loss = float('inf')
    TRG_PAD_IDX = 0

    # generating equation
    print("loading trained model...")

    if not torch.cuda.is_available():   # for CPU only
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))   # for GPUs

    return evaluate(model, vocab, imagetensor, device)


# Create a web app using FastAPI

app = FastAPI()

@app.put("/get_mml/")
async def upload_file(file: bytes = File()):
    '''
    Creates a web app using FastAPI for rendering MathML for an image
    by specifying the  parameters for the render_mml function.

    '''
    # convert png image to tensor
    i2t = Image2Tensor()
    imagetensor = i2t(file)


    # change the shape of tensor from (C_in, H, W)
    # to (1, C_in, H, w) [batch =1]
    imagetensor = imagetensor.unsqueeze(0)

    # read config file
    config_path = "ourmml_lstm_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    # read vocab.txt
    vocab = open("vocab.txt").readlines()

    model_path = "opennmt_ourmml_100K_lte100_best.pt"

    return render_mml(config, model_path, vocab, imagetensor)
