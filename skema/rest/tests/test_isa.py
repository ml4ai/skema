import json

from fastapi.testclient import TestClient
from skema.isa.isa_service import app
import skema.isa.data as isa_data
import pytest

client = TestClient(app)


@pytest.mark.ci_only
def test_align_eqns():
    """Test case for /align-eqns endpoint."""

    halfar_dome_eqn = isa_data.mml
    mention_json1_content = ""
    mention_json2_content = ""
    data = {
        "mml1": halfar_dome_eqn,
        "mml2": halfar_dome_eqn,
        "mention_json1": mention_json1_content,
        "mention_json2": mention_json2_content,
    }

    endpoint = "/isa/align-eqns"
    response = client.post(endpoint, params=data)
    expected = isa_data.expected

    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response of matching_ratio
    assert (
        json.loads(response.text)["matching_ratio"] == 1.0
    ), f"Response should be 1.0, but instead received {response.text}"
    # check response of union_graph
    assert (
        json.loads(response.text)["union_graph"] == expected
    ), f"Response should be {expected}, but instead received {response.text}"
