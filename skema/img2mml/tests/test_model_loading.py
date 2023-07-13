from pathlib import Path
from skema.img2mml.api import get_mathml_from_bytes, retrieve_model, Image2MathML
import os


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
    cwd = Path(__file__).parents[0].parents[0]
    config_path = cwd / "configs" / "xfmer_mml_config.json"
    vocab_path = (
        cwd / "trained_models" / "arxiv_im2mml_with_fonts_with_boldface_vocab.txt"
    )
    model_path = (
        cwd / "trained_models" / "cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt"
    )

    image2mathml_db = Image2MathML(
        config_path=config_path, vocab_path=vocab_path, model_path=model_path
    )
    assert image2mathml_db.model != None, "Fail to load the model checkpoint"
    assert image2mathml_db.vocab != None, "Fail to load the vocabulary file"
    assert image2mathml_db.config != None, "Fail to load the configuration file"
    return image2mathml_db


def test_local_loading_prediction():
    """Tests model loading and prediction"""
    # a) Local loading test
    image2mathml_db = local_loading()
    # b) Prediction test
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"
    with Path(image_path).open("rb") as infile:
        img_bytes = infile.read()

    try:
        mathml = get_mathml_from_bytes(img_bytes, image2mathml_db)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state dictionary file not found")
    except RuntimeError:
        raise RuntimeError(f"Error loading state dictionary from file")
    except Exception as e:
        raise Exception(f"Error converting the image: {e}")
    assert mathml is not None, "model failed to generate mml from image"