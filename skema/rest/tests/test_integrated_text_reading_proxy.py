from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import status

from skema.rest.integrated_text_reading_proxy import app, cosmos_client
from skema.rest.schema import MiraGroundingOutputItem

client = TestClient(app)


# TODO Add a unit test with a small document


# Test the cosmos endpoint
def test_cosmos():
    """Test that we are able to fetch COSMOS data correctly"""
    path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "CHIME_SVIIvR_model.pdf"
    with path.open("rb") as pdf:
        ret = cosmos_client(path.name, pdf)
    assert ret is not None and len(ret) > 0


def test_mira_grounding():
    """Test that we are getting grounding for entities"""
    queries = {"queries": ["infected", "suceptible"]}
    params = {"k": 5}
    ret = client.post("/ground_to_mira", params=params, json=queries)

    assert ret.status_code == status.HTTP_200_OK

    data = [[MiraGroundingOutputItem(**r) for r in q] for q in ret.json()]
    assert len(data) == 2, "Service didn't return results for all queries"
    assert all(len(groundings) == params["k"] for groundings in data), "Service didn't return the requested number of candidates for each query"



def test_healthcheck():
    """Test case for /healthcheck endpoint."""
    response = client.get("/healthcheck")
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    }
