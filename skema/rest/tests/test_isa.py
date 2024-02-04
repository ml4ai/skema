import json

from fastapi.testclient import TestClient
from skema.rest.workflows import app
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
    response = client.post(endpoint, json=single_snippet_payload)
    expected = """
    {"2":[0.3,7.0,["β*I*S/N-γ*I","D(1, t)(I)","β*I*S/N","β","I","S","N","γ*I","γ"],["gamma*i+r","D(1, t)(r)","gamma*i","gamma","i","r"],[0.0,1.0,-1.0,-1.0,4.0,-1.0,5.0,2.0,3.0],[0.0,1.0,7.0,8.0,4.0,6.0,-1.0,-1.0,-1.0],"digraph G {\n0 [color=blue, label=\"β*I*S/N-γ*I <<|>> gamma*i+r\"];\n1 [color=blue, label=\"D(1, t)(I) <<|>> D(1, t)(r)\"];\n2 [color=red, label=\"β*I*S/N\"];\n3 [color=red, label=\"β\"];\n4 [color=blue, label=\"I\"];\n5 [color=red, label=\"S\"];\n6 [color=blue, label=\"N <<|>> r\"];\n7 [color=blue, label=\"γ*I <<|>> gamma*i\"];\n8 [color=blue, label=\"γ <<|>> gamma\"];\n1 -> 0  [color=blue, label=\"=\"];\n2 -> 0  [color=red, label=\"+\"];\n3 -> 2  [color=red, label=\"*\"];\n4 -> 2  [color=red, label=\"*\"];\n5 -> 2  [color=red, label=\"*\"];\n6 -> 2  [color=red, label=\"/\"];\n7 -> 0  [color=red, label=\"-\"];\n8 -> 7  [color=blue, label=\"*\"];\n4 -> 7  [color=blue, label=\"*\"];\n7 -> 0  [color=green, label=\"+\"];\n6 -> 0  [color=green, label=\"+\"];\n}\n",[1,0,1,1,1,1,1,1,0]],"1":[0.8,2.0,["β*I*S/N-γ*I","D(1, t)(I)","β*I*S/N","β","I","S","N","γ*I","γ"],["i*beta*s-gamma*i+i","D(1, t)(i)","i*beta*s","i","beta","s","gamma*i","gamma"],[0.0,1.0,2.0,4.0,3.0,5.0,-1.0,6.0,7.0],[0.0,1.0,2.0,4.0,3.0,5.0,7.0,8.0,-1.0],"digraph G {\n0 [color=blue, label=\"β*I*S/N-γ*I <<|>> i*beta*s-gamma*i+i\"];\n1 [color=blue, label=\"D(1, t)(I)\"];\n2 [color=blue, label=\"β*I*S/N <<|>> i*beta*s\"];\n3 [color=blue, label=\"β <<|>> beta\"];\n4 [color=blue, label=\"I\"];\n5 [color=blue, label=\"S\"];\n6 [color=red, label=\"N\"];\n7 [color=blue, label=\"γ*I <<|>> gamma*i\"];\n8 [color=blue, label=\"γ <<|>> gamma\"];\n1 -> 0  [color=blue, label=\"=\"];\n2 -> 0  [color=blue, label=\"+\"];\n3 -> 2  [color=blue, label=\"*\"];\n4 -> 2  [color=blue, label=\"*\"];\n5 -> 2  [color=blue, label=\"*\"];\n6 -> 2  [color=red, label=\"/\"];\n7 -> 0  [color=blue, label=\"-\"];\n8 -> 7  [color=blue, label=\"*\"];\n4 -> 7  [color=blue, label=\"*\"];\n4 -> 0  [color=green, label=\"+\"];\n}\n",[1,0,1,0,1,0,1,0,0]],"3":[0.36,7.0,["β*I*S/N-γ*I","D(1, t)(I)","β*I*S/N","β","I","S","N","γ*I","γ"],["i*-(beta)*s+s","D(1, t)(s)","i*-(beta)*s","i","-(beta)","beta","s"],[0.0,1.0,2.0,5.0,3.0,6.0,4.0,-1.0,-1.0],[0.0,1.0,2.0,4.0,6.0,3.0,5.0,-1.0,-1.0],"digraph G {\n0 [color=blue, label=\"β*I*S/N-γ*I <<|>> i*-(beta)*s+s\"];\n1 [color=blue, label=\"D(1, t)(I) <<|>> D(1, t)(s)\"];\n2 [color=blue, label=\"β*I*S/N <<|>> i*-(beta)*s\"];\n3 [color=blue, label=\"β <<|>> beta\"];\n4 [color=blue, label=\"I\"];\n5 [color=blue, label=\"S\"];\n6 [color=blue, label=\"N <<|>> -(beta)\"];\n7 [color=red, label=\"γ*I\"];\n8 [color=red, label=\"γ\"];\n1 -> 0  [color=blue, label=\"=\"];\n2 -> 0  [color=blue, label=\"+\"];\n3 -> 2  [color=red, label=\"*\"];\n4 -> 2  [color=blue, label=\"*\"];\n5 -> 2  [color=blue, label=\"*\"];\n6 -> 2  [color=red, label=\"/\"];\n7 -> 0  [color=red, label=\"-\"];\n8 -> 7  [color=red, label=\"*\"];\n4 -> 7  [color=red, label=\"*\"];\n6 -> 2  [color=green, label=\"*\"];\n3 -> 6  [color=green, label=\"-\"];\n5 -> 0  [color=green, label=\"+\"];\n}\n",[1,0,1,1,1,1,1,1,1]]}
    """

    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response of matching_ratio
    assert (
        json.loads(response.text) == expected
    ), f"Response should be {expected}, but instead received {response.text}"