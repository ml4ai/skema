from pathlib import Path
from skema.img2mml.api import get_mathml_from_file, retrieve_model
import requests
import os


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


def test_img2mml_service():
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"
    SKEMA_EQ2MML_SERVICE = os.environ.get("SKEMA_EQMML_ADDRESS", "http://eqn2mml:8001")

    files = {
        "data": open(str(image_path), "rb"),
    }
    try:
        r = requests.post("{}/image/mml".format(SKEMA_EQ2MML_SERVICE), files=files)
        mathml = r.text
    except Exception as e:
        raise Exception(f"Error calling the img2mml service: {e}")
    assert mathml is not None
    print("The img2mml service is running well on server.")


if __name__ == "__main__":
    test_model_retrieval()
    test_local_loading_and_prediction()
    test_img2mml_service()