import json

from fastapi.testclient import TestClient
from skema.isa.isa_service import app
from skema.rest.workflows import app as workflow_app
import skema.isa.data as isa_data
import pytest

client = TestClient(app)
workflow_client = TestClient(workflow_app)

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
        str(json.loads(response.text)["union_graph"]) == expected
    ), f"Response should be {expected}, but instead received {response.text}"


@pytest.mark.ci_only
def test_align_code_eqn():
    """Test case for /isa/code-eqn-align endpoint."""

    single_snippet_payload = {
        "system": {
            "files": ["code.py"],
            "blobs": [
                'def sir(\n    s: float, i: float, r: float, beta: float, gamma: float, n: float\n) -> Tuple[float, float, float]:\n    """The SIR model, one time step."""\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale'
            ],
        },
        "mml": """<math>
      <mfrac>
        <mrow>
          <mi>d</mi>
          <mi>I</mi>
        </mrow>
        <mrow>
          <mi>d</mi>
          <mi>t</mi>
        </mrow>
      </mfrac>
      <mo>=</mo>
      <mfrac>
        <mrow>
            <mi>&#x03B2;</mi>
            <mi>I</mi>
            <mi>S</mi>
        </mrow>
        <mi>N</mi>
      </mfrac>
      <mo>&#x2212;</mo>
      <mi>&#x03B3;</mi>
      <mi>I</mi>
    </math>""",
    }

    endpoint = "/isa/code-eqn-align"
    response = workflow_client.post(endpoint, json=single_snippet_payload)

    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response of matching_ratio
    assert (
        json.loads(response.text)["1"][0] == 0.8
    ), f"Matching ratio should be 0.8, but instead received {json.loads(response.text)}"