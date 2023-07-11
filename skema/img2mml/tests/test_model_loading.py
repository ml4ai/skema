from pathlib import Path
from skema.img2mml.api import get_mathml_from_file, retrieve_model
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


def test_local_loading_and_prediction():
    """Tests model loading and prediction"""
    # FIXME: this should be split into two tests: a) one that instantiates the model and b) one that tests the instantiated model's ability to make predictions
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"

    try:
        mathml = get_mathml_from_file(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state dictionary file not found")
    except RuntimeError:
        raise RuntimeError(f"Error loading state dictionary from file")
    except Exception as e:
        raise Exception(f"Error converting the image: {e}")
    assert mathml is not None, "model failed to generate mml from image"