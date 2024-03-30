from pathlib import Path

from fastapi import status
from fastapi.testclient import TestClient
from pytest import approx

from skema.rest.integrated_text_reading_proxy import app
from skema.rest.schema import MiraGroundingOutputItem, TextReadingAnnotationsOutput

client = TestClient(app)


def test_text_integrated_extractions():
    """ Tests the integrated text extractions endpoint """
    # Read an example document to annotate
    params = {
        "annotate_skema": True,
        "annotate_mit": False
    }

    payload = {
        "texts": [
            "x = 0",
            "y = 1",
            "I: Infected population"
        ],
        "amrs": []
    }

    response = client.post(f"/integrated-text-extractions", params=params, json=payload)
    assert response.status_code == 200

    results = TextReadingAnnotationsOutput(**response.json())
    assert len(results.outputs) == 3, "One of the inputs doesn't have outputs"
    assert results.generalized_errors is None, f"Generalized TR errors"
    for ix, output in enumerate(results.outputs):
        assert output.data is not None, f"Document {ix + 1} didn't generate AttributeCollection"
        assert len(output.data.attributes) > 0, f"Document {ix + 1} generated an empty attribute collection"
        assert output.errors is None, f"Document {ix + 1} reported errors"


## EN: Comment this out until we can mock the cosmos endpoint to decouple our unit test from the status of their service
def test_integrated_pdf_extraction():
    """ Tests the pdf endpoint """
    params = {
        "annotate_skema": True,
        "annotate_mit": False
    }

    path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "CHIME_SVIIvR_model.pdf"
    with path.open("rb") as pdf:
        files = [
            ("pdfs", ("CHIME_SVIIvR_model.pdf", pdf, "application/pdf"))
        ]

        response = client.post(f"/integrated-pdf-extractions", params=params, files=files)

    assert response.status_code == 200

    results = TextReadingAnnotationsOutput(**response.json())
    assert len(results.outputs) == 1, "The inputs doesn't have outputs"
    assert results.generalized_errors is None, f"Generalized TR errors"
    for ix, output in enumerate(results.outputs):
        assert output.data is not None, f"Document {ix + 1} didn't generate AttributeCollection"
        assert len(output.data.attributes) > 0, f"Document {ix + 1} generated an empty attribute collection"
        assert output.errors is None, f"Document {ix + 1} reported errors"


# Test the cosmos endpoint
# EN: Commented this out as we don't control it (UWisc)
# def test_cosmos():
#     """Test that we are able to fetch COSMOS data correctly"""
#     path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "CHIME_SVIIvR_model.pdf"
#     with path.open("rb") as pdf:
#         ret = cosmos_client(path.name, pdf)
#     assert ret is not None and len(ret) > 0


def test_mira_grounding():
    """Test that we are getting grounding for entities"""
    queries = {"queries": ["infected", "suceptible"]}
    params = {"k": 5}
    ret = client.post("/ground_to_mira", params=params, json=queries)

    assert ret.status_code == status.HTTP_200_OK

    data = [[MiraGroundingOutputItem(**r) for r in q] for q in ret.json()]
    assert len(data) == 2, "Service didn't return results for all queries"
    assert all(len(groundings) == params["k"] for groundings in
               data), "Service didn't return the requested number of candidates for each query"


def test_extraction_evaluation():
    """ Test the extraction evaluation endpoint such that:
        - Runs end to end
        - Doesn't drastically change in performance due to a bug on the evaluation function
    """

    extractions_path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "eval" / "extractions.json"
    annotations_path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "eval" / "annotations.json"
    json_path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "eval" / "contents.json"

    with extractions_path.open("rb") as extractions, annotations_path.open("rb") as annotations, json_path.open(
            "rb") as json:
        files = {
            "extractions_file": ("paper_variable_extractions.json", extractions),
            "gt_annotations": ("paper_gt_annotations.json", annotations),
            "json_text": ("paper_cosmos_output.json", json),
        }

        response = client.post(f"/eval", files=files)

    assert response.status_code == status.HTTP_200_OK

    results = response.json()

    assert results['num_manual_annotations'] == 220, "There should be 220 gt manual annotations"
    assert results['precision'] == approx(0.5230769230768426), "Precision drastically different from the expected value"
    assert results['recall'] == approx(0.154545454545454542), "Recall drastically different from the expected value"
    assert results['f1'] == approx(0.23859649119285095), "F1 drastically different from the expected value"


def test_healthcheck():
    """Test case for /healthcheck endpoint."""
    response = client.get("/healthcheck")
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    }
