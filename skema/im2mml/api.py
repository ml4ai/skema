import json
from skema.im2mml.translate import Image2Tensor, render_mml

def get_mathml_from_bytes(data: bytes):
    # convert png image to tensor
    i2t = Image2Tensor()
    imagetensor = i2t(data)
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

def get_mathml_from_file(filepath):
    # convert png image to tensor

    with open(filepath, "rb") as f:
        data = f.read()

    i2t = Image2Tensor()
    imagetensor = i2t(data)
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
