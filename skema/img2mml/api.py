import json
from pathlib import Path
from skema.img2mml.translate import convert_to_torch_tensor, render_mml


def get_mathml_from_bytes(data: bytes):
    # read config file
    cwd = Path(__file__).parents[0]
    config_path = cwd / "configs/xfmer_mml_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    # convert png image to tensor
    imagetensor = convert_to_torch_tensor(data, config)

    # change the shape of tensor from (C_in, H, W)
    # to (1, C_in, H, w) [batch =1]
    imagetensor = imagetensor.unsqueeze(0)

    # read vocab.txt
    with open(cwd / "vocab.txt") as f:
        vocab = f.readlines()

    model_path = cwd / "trained_models/cnn_xfmer_omml-100K_best.pt"

    return render_mml(config, model_path, vocab, imagetensor)


def get_mathml_from_file(filepath) -> str:
    """Read an equation image file and convert it to MathML"""

    with open(filepath, "rb") as f:
        data = f.read()

    return get_mathml_from_bytes(data)
