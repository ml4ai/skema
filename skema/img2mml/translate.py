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
    """Pad image."""
    right, left, top, bottom = 8, 8, 8, 8
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height))
    result.paste(image, (left, top))
    return result

def convert_to_torch_tensor(image: bytes) -> torch.Tensor:
    """Convert image to torch tensor."""
    image = Image.open(io.BytesIO(image)).convert("L")
    image = image.resize((500, 50))
    image = pad_image(image)

    # convert to tensor
    image = transforms.ToTensor()(image)

    return image

def set_random_seed(seed: int) -> None:
    """Set up seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def define_model(
    config: dict, vocab: List[str], device: torch.device, model_type="xfmer"
) -> Image2MathML_Xfmer:
    """
    Defining the model
    initializing encoder, decoder, and model
    """

    print("Defining model...")

    model_type = config["model_type"]
    input_channels = config["input_channels"]
    output_dim = len(vocab)
    emb_dim = config["embedding_dim"]
    dec_hid_dim = config["decoder_hid_dim"]
    dropout = config["dropout"]
    max_len = config["max_len"]

    print(f"building {model_type} model...")

    dim_feedfwd = config["dim_feedforward_for_xfmer"]
    n_heads = config["n_xfmer_heads"]
    n_xfmer_encoder_layers = config["n_xfmer_encoder_layers"]
    n_xfmer_decoder_layers = config["n_xfmer_decoder_layers"]

    enc = {
        "CNN": CNN_Encoder(input_channels, dec_hid_dim, dropout, device),
        "XFMER": Transformer_Encoder(
            emb_dim,
            dec_hid_dim,
            n_heads,
            dropout,
            device,
            max_len,
            n_xfmer_encoder_layers,
            dim_feedfwd,
        ),
    }
    dec = Transformer_Decoder(
        emb_dim,
        n_heads,
        dec_hid_dim,
        output_dim,
        dropout,
        max_len,
        n_xfmer_decoder_layers,
        dim_feedfwd,
        device,
    )
    model = Image2MathML_Xfmer(enc, dec, vocab)

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
