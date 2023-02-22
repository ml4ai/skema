# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models.encoders.cnn_encoder import CNN_Encoder
from models.image2mml_xfmer import (
    Image2MathML_Xfmer,
)
from models.encoders.xfmer_encoder import Transformer_Encoder
from models.decoders.xfmer_decoder import Transformer_Decoder
import io
from typing import List
import logging
from logging import info

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)


def preprocess_img(image: Image.Image, config: dict) -> Image.Image:
    """ preprocessing image - cropping, resizing, and padding """

    # checking if the image lies wihin permissible boundary
    w, h = image.size
    max_h = config["max_input_hgt"]
    if h >= max_h:
        resize_factor = max_h/h

        # downsampling the image
        image = image.resize((int(image.size[0] * resize_factor),
                              int(image.size[1] * resize_factor,
                              Image.LANCZOS)))

    # converting to np array
    image_arr = np.asarray(image, dtype=np.uint8)
    # find where the data lies
    indices = np.where(image_arr != 255)
    # get the boundaries
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])

    # cropping tha image
    image =  image.crop(( x_min, y_min, x_max,  y_max ))

    # finding the bucket
    # [width, hgt, resize_factor]
    buckets = [
        [820,86,0.6],
        [615, 65,0.8],
        [492, 52,1],
        [410, 43,1.2],
        [350, 37,1.4]
        ]
    # current width, hgt
    crop_width, crop_hgt = image.size[0], image.size[1]

    # find correct bucket
    resize_factor = config["resizing_factor"]
    for b in buckets:
        w,h,r = b
        if crop_width <= w and crop_hgt <= h:
            resize_factor = r

    # resizing the image
    resize_factor = config["resizing_factor"]
    image = image.resize((int(image.size[0] * resize_factor),
                          int(image.size[1] * resize_factor)),
                          Image.LANCZOS)

    # padding
    pad = config["padding"]
    width = config["preprocessed_image_width"]
    height = config["preprocessed_image_height"]
    new_image = Image.new("RGB", (width, height), (255,255,255))
    new_image.paste(image, (pad,pad))

    return new_image


def convert_to_torch_tensor(image: bytes, config: dict) -> torch.Tensor:
    """Convert image to torch tensor."""
    image = Image.open(io.BytesIO(image)).convert("L")
    image = preprocess_img(image, config)

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
        "CNN": CNN_Encoder(
                        input_channels,
                        dec_hid_dim,
                        dropout,
                        device
                        ),

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
    model = Image2MathML_Xfmer(enc, dec, vocab, device)

    return model


def evaluate(
    model: Image2MathML_Xfmer,
    vocab_itos: dict,
    vocab_stoi:dict,
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
            img, device, is_inference=True,
            SOS_token=int(vocab_stoi["<sos>"]),
            EOS_token=int(vocab_stoi["<eos>"]),
            PAD_token=int(vocab_stoi["<pad>"]),
        )  # O: (1, max_len, output_dim), preds: (1, max_len)

        pred = list()
        for p in output:
            pred.append(vocab_itos[str(p)])

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

    # creating a dictionary from vocab list
    vocab_itos = dict()
    vocab_stoi = dict()
    for v in vocab:
        k, v = v.split()
        vocab_itos[v.strip()] = k.strip()
        vocab_stoi[k.strip()] = v.strip()

    # generating equation
    print("loading trained model...")

    # if state_dict keys has "module.<key_name>"
    # we need to remove the "module." from key_names
    if config["clean_state_dict"]:
        new_model = dict()
        for key,value in torch.load(model_path, map_location=torch.device('cpu')).items():
            new_model[key[7:]] = value
            model.load_state_dict(new_model, strict=False)

    else:
        if not torch.cuda.is_available():
            info("CUDA is not available, falling back to using the CPU.")
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(model_path))

    return evaluate(model, vocab_itos, vocab_stoi, imagetensor, device)
