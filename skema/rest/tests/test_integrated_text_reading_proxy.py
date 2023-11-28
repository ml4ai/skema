from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import status

from skema.rest.integrated_text_reading_proxy import app, cosmos_client
from skema.rest.schema import MiraGroundingOutputItem, TextReadingAnnotationsOutput

client = TestClient(app)


# def test_text_integrated_extractions():
#     """ Tests the integrated text extractions endpoint """
#     # Read an example document to annotate
#     params = {
#         "annotate_skema": True,
#         "annotate_mit": False
#     }
#
#     payload = {
#         "texts": [
#             "x = 0",
#             "y = 1",
#             "I: Infected population"
#         ]
#     }
#
#     response = client.post(f"/integrated-text-extractions", params=params, json=payload)
#     assert response.status_code == 200
#
#     results = TextReadingAnnotationsOutput(**response.json())
#     assert len(results.outputs) == 3, "One of the inputs doesn't have outputs"
#     assert results.generalized_errors is None, f"Generalized TR errors"
#     for ix, output in enumerate(results.outputs):
#         assert output.data is not None, f"Document {ix + 1} didn't generate AttributeCollection"
#         assert len(output.data.attributes) > 0, f"Document {ix + 1} generated an empty attribute collection"
#         assert output.errors is None, f"Document {ix + 1} reported errors"


## EN: Comment this out until we can mock the cosmos endpoint to decouple our unit test from the status of their service
# def test_integrated_pdf_extraction():
#     """ Tests the pdf endpoint """
#     params = {
#         "annotate_skema": True,
#         "annotate_mit": False
#     }
#
#     path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "CHIME_SVIIvR_model.pdf"
#     with path.open("rb") as pdf:
#         files = [
#             ("pdfs", ("CHIME_SVIIvR_model.pdf", pdf, "application/pdf"))
#         ]
#
#         response = client.post(f"/integrated-pdf-extractions", params=params, files=files)
#
#     assert response.status_code == 200
#
#     results = TextReadingAnnotationsOutput(**response.json())
#     assert len(results.outputs) == 1, "The inputs doesn't have outputs"
#     assert results.generalized_errors is None, f"Generalized TR errors"
#     for ix, output in enumerate(results.outputs):
#         assert output.data is not None, f"Document {ix + 1} didn't generate AttributeCollection"
#         assert len(output.data.attributes) > 0, f"Document {ix + 1} generated an empty attribute collection"
#         assert output.errors is None, f"Document {ix + 1} reported errors"


# Test the cosmos endpoint
# EN: Commented this out as we don't control it (UWisc)
# def test_cosmos():
#     """Test that we are able to fetch COSMOS data correctly"""
#     path = Path(__file__).parents[0] / "data" / "integrated_text_reading" / "CHIME_SVIIvR_model.pdf"
#     with path.open("rb") as pdf:
#         ret = cosmos_client(path.name, pdf)
#     assert ret is not None and len(ret) > 0


# def test_mira_grounding():
#     """Test that we are getting grounding for entities"""
#     queries = {"queries": ["infected", "suceptible"]}
#     params = {"k": 5}
#     ret = client.post("/ground_to_mira", params=params, json=queries)
#
#     assert ret.status_code == status.HTTP_200_OK
#
#     data = [[MiraGroundingOutputItem(**r) for r in q] for q in ret.json()]
#     assert len(data) == 2, "Service didn't return results for all queries"
#     assert all(len(groundings) == params["k"] for groundings in
#                data), "Service didn't return the requested number of candidates for each query"
#
#
# def test_healthcheck():
#     """Test case for /healthcheck endpoint."""
#     response = client.get("/healthcheck")
#     assert response.status_code in {
#         status.HTTP_200_OK,
#         status.HTTP_502_BAD_GATEWAY,
#         status.HTTP_500_INTERNAL_SERVER_ERROR
#     }
