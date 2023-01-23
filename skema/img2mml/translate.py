# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skema.img2mml.models.encoders.cnn_encoder import CNN_Encoder
from skema.img2mml.models.image2mml_xfmer_for_inference import (
    Image2MathML_Xfmer,
)
from skema.img2mml.models.encoders.xfmer_encoder import Transformer_Encoder
from skema.img2mml.models.decoders.xfmer_decoder import Transformer_Decoder
import io
from typing import List


def pad_image(image: Image.Image) -> Image.Image:
    right, left, top, bottom = 8, 8, 8, 8
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height))
    result.paste(image, (left, top))
    return result

def convert_to_torch_tensor(image: bytes) -> torch.Tensor:
    print("Converting image to torch tensor...")
    image = Image.open(io.BytesIO(image)).convert("L")
    image = image.resize((500, 50))
    image = pad_image(image)

    # convert to tensor
    image = transforms.ToTensor()(image)

    return image

def set_random_seed(seed: int) -> None:
    """
    Setting up seed
    """
    # set up seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def define_model(
    config: dict, VOCAB: List[str], DEVICE: torch.device, model_type="xfmer"
) -> Image2MathML_Xfmer:
    """
    Defining the model
    initializing encoder, decoder, and model
    """

    print("Defining model...")

    MODEL_TYPE = config["model_type"]
    INPUT_CHANNELS = config["input_channels"]
    OUTPUT_DIM = len(VOCAB)
    EMB_DIM = config["embedding_dim"]
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
    img: torch.Tensor,
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
    # set_random_seed
    set_random_seed(config["seed"])

    # defining model using DataParallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model: Image2MathML_Xfmer = define_model(config, vocab, device).to(device)

    # generating equation
    print("loading trained model...")

    if not torch.cuda.is_available():
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        model.load_state_dict(torch.load(model_path))

    return evaluate(model, vocab, imagetensor, device)
