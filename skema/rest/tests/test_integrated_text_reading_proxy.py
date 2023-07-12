from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import status

from skema.rest.integrated_text_reading_proxy import app, cosmos_client

client = TestClient(app)


# TODO Add a unit test with a small document


# Commented out for now
# Test the cosmos endpoint
# def test_cosmos():
#     """Test that we are able to fetch COSMOS data correctly"""
#     path = Path(__file__).parents[0] / "data" / "CHIME_SVIIvR_model.pdf"
#     with path.open("rb") as pdf:
#         ret = cosmos_client(path.name, pdf)
#     assert ret is not None and len(ret) > 0


def test_healthcheck():
    """Test case for /healthcheck endpoint."""
    response = client.get("/healthcheck")
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    }


if __name__ == "__main__":
    test_cosmos()
