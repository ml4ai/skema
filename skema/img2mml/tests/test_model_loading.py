from pathlib import Path
from skema.img2mml.api import (
    get_mathml_from_bytes,
    retrieve_model,
    load_vocab,
    load_model,
    check_gpu_availability,
)
import os
import json
from skema.img2mml.models.image2mml_xfmer import Image2MathML_Xfmer


def test_model_retrieval():
    """Tests model retrieval"""
    model_path = retrieve_model()
    # Delete the model file if it exists
    if Path(model_path).exists():
        os.remove(model_path)
    # Retrieve the model again
    model_path = retrieve_model()
    assert Path(model_path).exists(), f"model was not found at {model_path}"


def local_loading():
    """Tests local loading files"""
    # Read config file
    cwd = Path(__file__).parents[0].parents[0]
    config_path = cwd / "configs" / "xfmer_mml_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)
    assert config != None, "Fail to load the configuration file"
    #  Load the image2mathml vocabulary
    VOCAB_PATH = (
        cwd / "trained_models" / "arxiv_im2mml_with_fonts_with_boldface_vocab.txt"
    )
    vocab, vocab_itos, vocab_stoi = load_vocab(vocab_path=VOCAB_PATH)
    assert vocab != None, "Fail to load the vocabulary file"

    #  Load the image2mathml model
    model_path = (
        cwd / "trained_models" / "cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt"
    )
    MODEL_PATH = retrieve_model(model_path=model_path)
    device = check_gpu_availability()
    img2mml_model: Image2MathML_Xfmer = load_model(
        model_path=MODEL_PATH, config=config, vocab=vocab, device=device
    )
    assert img2mml_model != None, "Fail to load the model checkpoint"
    return img2mml_model, config, vocab_itos, vocab_stoi, device


def test_local_loading_prediction():
    """Tests model loading and prediction"""
    # a) Local loading test
    img2mml_model, config, vocab_itos, vocab_stoi, device = local_loading()
    # b) Prediction test
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"
    with Path(image_path).open("rb") as infile:
        img_bytes = infile.read()

    try:
        mathml = get_mathml_from_bytes(
            img_bytes, img2mml_model, config, vocab_itos, vocab_stoi, device
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state dictionary file not found")
    except RuntimeError:
        raise RuntimeError(f"Error loading state dictionary from file")
    except Exception as e:
        raise Exception(f"Error converting the image: {e}")
    assert mathml is not None, "model failed to generate mml from image"