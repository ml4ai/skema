# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skema.img2mml.models.encoders.cnn_encoder import CNN_Encoder
from skema.img2mml.models.image2mml_xfmer import Image2MathML_Xfmer
from skema.img2mml.models.encoders.xfmer_encoder import Transformer_Encoder
from skema.img2mml.models.decoders.xfmer_decoder import Transformer_Decoder
import io
from typing import List
import logging
from logging import info
import cv2
import re

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)


def remove_eqn_number(image: Image.Image, threshold: float = 0.1) -> Image.Image:
    """
    Remove equation number from an image of an equation.

    Args:
        image (Image.Image): The input image.
        threshold (float, optional): The threshold to determine the size of the equation number.
            A smaller threshold will consider larger areas as equation numbers.
            Defaults to 0.1.

    Returns:
        Image.Image: The modified image with the equation number removed.
    """
    image_arr = np.asarray(image, dtype=np.uint8)
    # Invert the image to make the blank regions black
    inverted = cv2.bitwise_not(image_arr)

    # Get the width and height of the image
    height, width = inverted.shape[:2]

    # Start scanning from the right side
    column_sum = np.sum(inverted, axis=0)
    rightmost_column = width - 1
    leftmost_column = rightmost_column
    while leftmost_column >= 0:
        if column_sum[leftmost_column] != 0:
            if rightmost_column - leftmost_column > threshold * width:
                image_arr = image_arr[:, 0:leftmost_column]
                return Image.fromarray(image_arr)

            leftmost_column -= 1
            rightmost_column = leftmost_column
        else:
            leftmost_column -= 1

    return Image.fromarray(image_arr)


def preprocess_img(image: Image.Image, config: dict) -> Image.Image:
    """preprocessing image - cropping, resizing, and padding"""
    # remove equation number if having
    image = remove_eqn_number(image)
    # checking if the image lies within permissible boundary
    w, h = image.size
    max_h = config["max_input_hgt"]
    if h >= max_h:
        resize_factor = max_h / h

        # downsampling the image
        image = image.resize(
            (
                int(image.size[0] * resize_factor),
                int(image.size[1] * resize_factor),
            ),
            Image.Resampling.LANCZOS,
        )

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
    image = image.crop((x_min, y_min, x_max, y_max))

    # finding the bucket
    # [width, hgt, resize_factor]
    # v1
    # buckets = [
    #     [820, 86, 0.6],
    #     [615, 65, 0.8],
    #     [492, 52, 1],
    #     [410, 43, 1.2],
    #     [350, 37, 1.4],
    # ]
    # v2 v3
    buckets = [
        [820, 86, 0.6],
        [703, 74, 0.7],
        [615, 65, 0.8],
        [547, 58, 0.9],
        [492, 52, 1],
        [447, 47, 1.1],
        [410, 43, 1.2],
        [379, 40, 1.3],
        [350, 37, 1.4],
        [328, 35, 1.5],
    ]
    # v4
    # buckets = [
    #     [878, 92, 0.56],
    #     [780, 82, 0.63],
    #     [683, 72, 0.72],
    #     [592, 62, 0.83],
    #     [492, 52, 1],
    #     [400, 42, 1.23],
    #     [302, 32, 1.625],
    #     [208, 22, 2.36],
    #     [113, 12, 4.33],
    # ]
    # current width, hgt
    crop_width, crop_hgt = image.size[0], image.size[1]

    # find correct bucket
    resize_factor = config["resizing_factor"]
    for b in buckets:
        w, h, r = b
        if crop_width <= w and crop_hgt <= h:
            resize_factor = r

    # resizing the image
    # resize_factor = config["resizing_factor"]
    image = image.resize(
        (
            int(image.size[0] * resize_factor),
            int(image.size[1] * resize_factor),
        ),
        Image.LANCZOS,
    )

    # padding
    pad = config["padding"]
    width = config["preprocessed_image_width"]
    height = config["preprocessed_image_height"]
    new_image = Image.new("RGB", (width, height), (255, 255, 255))
    new_image.paste(image, (pad, pad))

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
    len_dim = 930

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
            len_dim,
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


def add_semicolon_to_unicode(string: str) -> str:
    """
    Checks if the string contains Unicode starting with '&#x' and adds a semicolon ';' after it.
    Args:
        string: The input string to check.
    Returns:
        The modified string with semicolons added after Unicode.
    """
    # Define a regular expression pattern to match '&#x' followed by hexadecimal characters
    pattern = r"&#x[0-9A-Fa-f]+"

    # Find all matches in the string using the pattern
    matches = re.findall(pattern, string)

    # Iterate over the matches and add semicolon after each Unicode
    for match in matches:
        string = string.replace(match, match + ";")

    return string


def remove_spaces_between_tags(mathml_string: str) -> str:
    """
    Remove spaces between ">" and "<" in a MathML string.

    Args:
        mathml_string (str): The MathML string to process.

    Returns:
        str: The modified MathML string with spaces removed between tags.
    """
    pattern = r">(.*?)<"
    replaced_string = re.sub(
        pattern, lambda match: match.group(0).replace(" ", ""), mathml_string
    )
    return replaced_string


def evaluate(
    model: Image2MathML_Xfmer,
    vocab_itos: dict,
    vocab_stoi: dict,
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
            img,
            device,
            is_inference=True,
            SOS_token=int(vocab_stoi["<sos>"]),
            EOS_token=int(vocab_stoi["<eos>"]),
            PAD_token=int(vocab_stoi["<pad>"]),
        )  # O: (1, max_len, output_dim), preds: (1, max_len)

        pred = list()
        for p in output:
            pred.append(vocab_itos[str(p)])

        pred_seq = " ".join(pred[1:-1])
        return add_semicolon_to_unicode(remove_spaces_between_tags(pred_seq))


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
        for key, value in torch.load(
            model_path, map_location=torch.device("cpu")
        ).items():
            new_model[key[7:]] = value
            model.load_state_dict(new_model, strict=False)

    else:
        if not torch.cuda.is_available():
            info("CUDA is not available, falling back to using the CPU.")
            new_model = dict()
            for key, value in torch.load(
                model_path, map_location=torch.device("cpu")
            ).items():
                new_model[key[7:]] = value
                model.load_state_dict(new_model, strict=False)
        else:
            model.load_state_dict(torch.load(model_path))

    return evaluate(model, vocab_itos, vocab_stoi, imagetensor, device)
