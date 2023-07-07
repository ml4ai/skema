import json
import os
import requests
from pathlib import Path

from skema.rest.proxies import SKEMA_MATHJAX_ADDRESS
from skema.img2mml.translate import convert_to_torch_tensor, render_mml


def get_mathml_from_bytes(data: bytes):
    # read config file
    cwd = Path(__file__).parents[0]
    config_path = cwd / "configs" / "xfmer_mml_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)
    # convert png image to tensor
    imagetensor = convert_to_torch_tensor(data, config)

    # change the shape of tensor from (C_in, H, W)
    # to (1, C_in, H, w) [batch =1]
    imagetensor = imagetensor.unsqueeze(0)

    MODEL_BASE_ADDRESS = "https://artifacts.askem.lum.ai/skema/img2mml/models"
    MODEL_NAME = "cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt"
    VOCAB_NAME = "arxiv_im2mml_with_fonts_with_boldface_vocab.txt"

    # read vocab.txt
    with open(cwd / "trained_models" / VOCAB_NAME) as f:
        vocab = f.readlines()

    # Construct the full path for the model file
    model_path = cwd / "trained_models" / MODEL_NAME

    import urllib.request

    # Check if the model file already exists
    if not os.path.exists(model_path):
        # If the file doesn't exist, download it from the specified URL
        url = f"{MODEL_BASE_ADDRESS}/{MODEL_NAME}"
        print(url)
        print("Downloading the model checkpoint...")
        urllib.request.urlretrieve(url, model_path)

    return render_mml(config, model_path, vocab, imagetensor)


def get_mathml_from_file(filepath) -> str:
    """Read an equation image file and convert it to MathML"""

    with open(filepath, "rb") as f:
        data = f.read()

    return get_mathml_from_bytes(data)


def get_mathml_from_latex(eqn: str) -> str:
    """Read a LaTeX equation string and convert it to presentation MathML"""

    # Define the webservice address from the MathJAX service
    webservice = SKEMA_MATHJAX_ADDRESS
    print(f"Connecting to {webservice}")

    # Translate and save each LaTeX string using the NodeJS service for MathJax
    res = requests.post(
        f"{webservice}/tex2mml",
        headers={"Content-type": "application/json"},
        json={"tex_src": eqn},
    )
    if res.status_code == 200:
        return res.text
    else:
        try:
            res.raise_for_status()
        except requests.HTTPError as e:
            return f"HTTP error occurred: {e}"
        except requests.ConnectionError as e:
            return f"Connection error occurred: {e}"
        except requests.Timeout as e:
            return f"Timeout error occurred: {e}"
        except requests.RequestException as e:
            return f"An error occurred: {e}"
        finally:
            return "Conversion Failed."
