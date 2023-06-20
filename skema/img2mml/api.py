import json
from pathlib import Path
import requests
import re
from skema.img2mml.translate import convert_to_torch_tensor, render_mml


def get_mathml_from_bytes(data: bytes):
    # convert png image to tensor
    imagetensor = convert_to_torch_tensor(data)

    # change the shape of tensor from (C_in, H, W)
    # to (1, C_in, H, w) [batch =1]
    imagetensor = imagetensor.unsqueeze(0)

    # read config file
    cwd = Path(__file__).parents[0]
    config_path = cwd / "configs" / "ourmml_xfmer_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    # read vocab.txt
    with open(cwd / "vocab.txt") as f:
        vocab = f.readlines()

    model_path = cwd / "trained_models" / "cnn_xfmer_OMML-90K_best_model_RPimage.pt"

    return render_mml(config, model_path, vocab, imagetensor)


def get_mathml_from_file(filepath) -> str:
    """Read an equation image file and convert it to MathML"""

    with open(filepath, "rb") as f:
        data = f.read()

    return get_mathml_from_bytes(data)


def get_mathml_from_latex(eqn) -> str:
    """Read a LaTeX equation string and convert it to presentation MathML"""

    # Define the webservice address from the MathJAX service
    # FIXME: this should be set using an ENV variable
    webservice = "http://localhost:8081"
    # Translate and save each LaTeX string using the NodeJS service for MathJax
    res = requests.post(
        f"{webservice}/tex2mml",
        headers={"Content-type": "application/json"},
        json={"tex_src": json.dumps(eqn)},
    )

    if res.status_code == 200:
        clean_res = (
            res.content.decode("utf-8")[1:-1]
            .replace("\\n", "")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
            .strip()
        )
        clean_res = re.sub(r"\s+", " ", clean_res)
        return clean_res
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
