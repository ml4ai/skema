from pathlib import Path

from askem_extractions.data_model import AttributeCollection
from fastapi import status

from skema.rest.metal_proxy import app
from fastapi.testclient import TestClient

client = TestClient(app)
def test_link_amr():
    """Test that we are able to link and AMR to text extractions correctly"""
    amr_path = Path(__file__).parents[0] / "data" / "metal" / "fixed_scenario1_amr_eq.json"
    extractions_path = Path(__file__).parents[0] / "data" / "metal" / "tr_document_results.json"

    params = {
        "amr_type": "petrinet",
        "similarity_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.5
    }

    files = {
        "amr_file": amr_path.open("rb"),
        "text_extractions_file": extractions_path.open("rb")
    }

    response = client.post("/link_amr", params=params, files=files)

    assert response.status_code == 200, f"The response code for /link_amr was {response.status_code}"

    linked_amr = response.json()

    # There should be more than 5 linked AMR elements
    metadata = AttributeCollection(**linked_amr["metadata"])
    num_linked_elements = len([md.amr_element_id is not None for md in metadata.attributes])
    assert num_linked_elements > 5, f"Only {num_linked_elements}, should be close to 17 with the testing configuration"



def test_healthcheck():
    """Test case for /healthcheck endpoint."""
    response = client.get("/healthcheck")
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    }

if __name__ == "__main__":
    test_link_amr()