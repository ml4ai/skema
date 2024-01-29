from httpx import AsyncClient
from skema.rest.workflows import app
import pytest
import json


@pytest.mark.ci_only
@pytest.mark.asyncio
async def test_post_latex_to_decapodes():
    """
    Test case for /latex/equations-to-decapodes endpoint.
    """
    equations = [
        "\\frac{d H}{dt}=\\nabla \\cdot {(\\Gamma*H^{n+2}*\\left|\\nabla{H}\\right|^{n-1}*\\nabla{H})}"
    ]

    endpoint = "/latex/equations_to_decapodes"

    async with AsyncClient(app=app, base_url="http://latex-to-decapodes-test") as ac:
        response = await ac.put(endpoint, json=equations)
    expected = '{"Var":[{"type":"infer","name":"•1"},{"type":"infer","name":"mult_1"},{"type":"infer","name":"mult_2"},{"type":"infer","name":"mult_3"},{"type":"infer","name":"Γ"},{"type":"infer","name":"•2"},{"type":"infer","name":"H"},{"type":"infer","name":"sum_1"},{"type":"infer","name":"n"},{"type":"Literal","name":"2"},{"type":"infer","name":"•3"},{"type":"infer","name":"•4"},{"type":"infer","name":"•5"},{"type":"infer","name":"•6"},{"type":"Literal","name":"1"},{"type":"infer","name":"•7"},{"type":"infer","name":"•8"}],"Op1":[{"src":7,"tgt":13,"op1":"Grad"},{"src":13,"tgt":12,"op1":"Abs"},{"src":7,"tgt":16,"op1":"Grad"},{"src":2,"tgt":1,"op1":"Div"},{"src":7,"tgt":17,"op1":"D(1,t)"}],"Op2":[{"proj1":7,"proj2":8,"res":6,"op2":"^"},{"proj1":5,"proj2":6,"res":4,"op2":"*"},{"proj1":9,"proj2":15,"res":14,"op2":"-"},{"proj1":12,"proj2":14,"res":11,"op2":"^"},{"proj1":4,"proj2":11,"res":3,"op2":"*"},{"proj1":3,"proj2":16,"res":2,"op2":"*"}],"Σ":[{"sum":8}],"Summand":[{"summand":9,"summation":1},{"summand":10,"summation":1}]}'
    # check for route's existence
    assert (
        any(route.path == endpoint for route in app.routes) == True
    ), "{endpoint} does not exist for app"
    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response
    assert (
        json.loads(response.text) == expected
    ), f"Response should be {expected}, but instead received {response.text}"