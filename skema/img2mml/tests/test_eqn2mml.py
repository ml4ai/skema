from pathlib import Path
from fastapi.testclient import TestClient
from skema.img2mml.eqn2mml import app
from skema.img2mml.api import retrieve_model

import pytest


client = TestClient(app)

@pytest.fixture
def retrieve_and_load_model():
    '''Retrieves model if not present and stores on disk'''
    retrieve_model()
    
def test_post_image_to_mml():
    '''Test case for /image/mml endpoint.'''
    retrieve_and_load_model
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "261.png"
    with open(image_path, "rb") as infile:
        image_bytes = infile.read()
    response = client.post("/image/mml", files={"data": image_bytes})
    assert response.status_code == 200, "Request was unsuccessful"
    assert response.text != None, "Response was empty"

def test_healthcheck():
    '''Test case for /img2mml/healthcheck endpoint.'''
    response = client.get("/img2mml/healthcheck")
    assert response.status_code == 200