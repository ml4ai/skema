from fastapi.testclient import TestClient

from skema.rest.integrated_text_reading_proxy import app

client = TestClient(app)


# TODO Add a unit test with a small document

def test_healthcheck():
    '''Test case for /healthcheck endpoint.'''
    response = client.get("/healthcheck")
    assert response.status_code == 200
