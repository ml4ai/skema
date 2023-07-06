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


def calculate_scale_factor(
    image: Image.Image, target_width: int, target_height: int
) -> float:
    """
    Calculate the scale factor to normalize the input image to the target width and height while preserving the
    original aspect ratio. If the original aspect ratio is larger than the target aspect ratio, the scale factor
    will be calculated based on width. Otherwise, it will be calculated based on height.

    Args:
        image (PIL.Image.Image): The input image to be normalized.
        target_width (int): The target width for normalization.
        target_height (int): The target height for normalization.

    Returns:
        float: The scale factor to normalize the image.
    """
    original_width, original_height = image.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        # Calculate scale factor based on width
        scale_factor = target_width / original_width
    else:
        # Calculate scale factor based on height
        scale_factor = target_height / original_height

    return scale_factor


def preprocess_img(image: Image.Image, config: dict) -> Image.Image:
    """preprocessing image - cropping, resizing, and padding"""
    # remove equation number if having
    image = remove_eqn_number(image)

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

    # calculate the target width and height
    target_width = config["preprocessed_image_width"] - 2 * config["padding"]
    target_height = config["preprocessed_image_height"] - 2 * config["padding"]
    # calculate the scale factor
    resize_factor = calculate_scale_factor(image, target_width, target_height)

    # resizing the image
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
    len_dim = 2500

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
    Checks if the string contains Unicode starting with '&#x' and adds a semicolon ';' after each occurrence if missing.

    Args:
        string (str): The input string to check.

    Returns:
        str: The modified string with semicolons added after each Unicode occurrence if necessary.
    """
    # Define a regular expression pattern to match '&#x' followed by hexadecimal characters
    pattern = r"&#x[0-9A-Fa-f]+"

    def add_semicolon(match):
        unicode_value = match.group(0)
        if not unicode_value.endswith(";"):
            unicode_value += ";"
        return unicode_value

    # Find all matches in the string using the pattern and process each match individually
    modified_string = re.sub(pattern, add_semicolon, string)

    return modified_string


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
