from pathlib import Path
from skema.img2mml.api import get_mathml_from_file, retrieve_model
import requests


def test_model_retrieval():
    model_path = retrieve_model()
    assert Path(model_path).exists()
    print("The model checkpoint is existed.")


def test_local_loading_and_prediction():
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
    assert mathml is not None
    print("The img2mml model is running well locally.")


def get_mml(image_path: str, url: str) -> str:
    """
    It sends the http requests to put in an image to translate it into MathML.
    """
    with open(image_path, "rb") as f:
        r = requests.put(url, files={"file": f})
    return r.text


def test_img2mml_service():
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"
    url = "http://localhost:8000/get-mml"
    try:
        mathml = get_mml(str(image_path), url)
    except Exception as e:
        raise Exception(f"Error calling the img2mml service: {e}")
    assert mathml is not None
    print("The img2mml service is running well on server.")


if __name__ == "__main__":
    test_model_retrieval()
    test_local_loading_and_prediction()
    test_img2mml_service()