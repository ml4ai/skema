# -*- coding: utf-8 -*-

import os, subprocess
import random
import numpy as np
import time
import json
import math
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms
from PIL import Image
from models.encoders.cnn_encoder import CNN_Encoder
from models.image2mml_xfmer_for_inference import Image2MathML_Xfmer
from models.encoders.xfmer_encoder import Transformer_Encoder
from models.decoders.xfmer_decoder import Transformer_Decoder
import io
from typing import List
from fastapi import FastAPI, File


class Image2Tensor(object):
    """
    This class takes in an image and generates a tensor object
    """

    def __init__(self):
        print("Converting image to torch tensor...")

    def crop_image(self, image, size):
        return transforms.functional.crop(image, 0, 0, size[0], size[1])

    def resize_image(self, image):
        return image.resize((500, 50))

    def pad_image(self, IMAGE):
        right, left, top, bottom = 8, 8, 8, 8
        width, height = IMAGE.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(IMAGE.mode, (new_width, new_height))
        result.paste(IMAGE, (left, top))
        return result

    def __call__(self, image_path):
        """
        convert png image to tensor

        :params: png image path
        :return: processed image tensor
        """

        # crop, resize, and pad the image
        IMAGE = Image.open(image_path).convert("L")
        IMAGE = self.resize_image(IMAGE)
        IMAGE = self.pad_image(IMAGE)
        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

        return IMAGE


def set_random_seed(SEED: int) -> None:
    """
    setting up seed
    """
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def define_model(
    config: dict, VOCAB: List[str], DEVICE: torch.device, model_type="xfmer"
) -> Image2MathML_Xfmer:
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
    ENCODING_TYPE = config["encoding_type"]
    DEC_HID_DIM = config["decoder_hid_dim"]
    DROPOUT = config["dropout"]
    MAX_LEN = config["max_len"]

    print(f"building {MODEL_TYPE} model...")

    DIM_FEEDFWD = config["dim_feedforward_for_xfmer"]
    N_HEADS = config["n_xfmer_heads"]
    N_XFMER_ENCODER_LAYERS = config["n_xfmer_encoder_layers"]
    N_XFMER_DECODER_LAYERS = config["n_xfmer_decoder_layers"]

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
    model = Image2MathML_Xfmer(ENC, DEC, VOCAB)

    return model


def evaluate(
    model: Image2MathML_Xfmer,
    vocab: List[str],
    img: Image2Tensor,
    device: torch.device,
) -> str:

    """
    It predicts the sequence for the image to translate it into MathML contents
    """

    model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = model(
            img, device, is_train=False, is_test=False
        )  # O: (B, max_len, output_dim), preds: (B, max_len)

        vocab_dict = {}
        for v in vocab:
            k, v = v.split()
            vocab_dict[v.strip()] = k.strip()

        pred = list()
        for p in output:
            pred.append(vocab_dict[str(p)])

        pred_seq = " ".join(pred[1:-1])
        return pred_seq


def render_mml(config: dict, model_path, vocab: List[str], imagetensor) -> str:

    """
    It allows us to obtain mathML for an image
    """
    # parameters
    optimizer_type = config["optimizer_type"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    (beta_1, beta_2) = config["beta_1"], config["beta_2"]
    CLIP = config["clip"]
    SEED = config["seed"]
    model_type = config["model_type"]
    load_trained_model = config["load_trained_model_for_testing"]
    use_single_gpu = config["use_single_gpu"]
    max_len = config["max_len"]

    # set_random_seed
    set_random_seed(SEED)

    # defining model using DataParallel
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model: Image2MathML_Xfmer = define_model(config, vocab, device).to(device)

    # optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta_1, beta_2),
        )

    best_valid_loss = float("inf")
    TRG_PAD_IDX = 0

    # generating equation
    print("loading trained model...")

    if not torch.cuda.is_available():
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        model.load_state_dict(torch.load(model_path))

    return evaluate(model, vocab, imagetensor, device)


# Create a web app using FastAPI

app = FastAPI()


@app.put("/get-mml", summary="Get MathML representation of an equation image")
async def get_mathml(file: bytes = File()):
    """
    Creates a web app using FastAPI for rendering MathML for an image
    by specifying the  parameters for the render_mml function.
    """
    # convert png image to tensor
    i2t = Image2Tensor()
    imagetensor = i2t(file)
    print("image done!")

    # change the shape of tensor from (C_in, H, W)
    # to (1, C_in, H, w) [batch =1]
    imagetensor = imagetensor.unsqueeze(0)

    # read config file
    config_path = "configs/ourmml_xfmer_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    # read vocab.txt
    vocab = open("vocab.txt").readlines()

    model_path = "trained_models/cnn_xfmer_OMML-90K_best_model_RPimage.pt"

    return render_mml(config, model_path, vocab, imagetensor)
