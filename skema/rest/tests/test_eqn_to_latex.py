import base64
from pathlib import Path
from httpx import AsyncClient
from skema.rest.workflows import app
import pytest
import json


@pytest.mark.ci_only
@pytest.mark.asyncio
async def test_post_image_to_latex():
    """Test case for /images/equations-to-latex endpoint."""

    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "img2latex" / "halfar.png"
    files = {
        "data": open(image_path, "rb"),
    }

    endpoint = "/images/equations-to-latex"
    # see https://fastapi.tiangolo.com/advanced/async-tests/#async-tests
    async with AsyncClient(app=app, base_url="http://eqn-to-latex-test") as ac:
        response = await ac.post(endpoint, files=files)
    expected = "\\frac{d H}{dt}=\\nabla \\cdot {(\\Gamma*H^{n+2}*\\left|\\nabla{H}\\right|^{n-1}*\\nabla{H})}"
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


@pytest.mark.ci_only
@pytest.mark.asyncio
async def test_post_image_to_latex_base64():
    """Test case for /images/base64/equations-to-latex endpoint."""
    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "img2latex" / "halfar.png"
    with Path(image_path).open("rb") as infile:
        img_bytes = infile.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    endpoint = "/images/base64/equations-to-latex"
    # see https://fastapi.tiangolo.com/advanced/async-tests/#async-tests
    async with AsyncClient(app=app, base_url="http://eqn-to-latex-base64-test") as ac:
        response = await ac.post(endpoint, data=img_b64)
    expected = "\\frac{d H}{dt}=\\nabla \\cdot {(\\Gamma*H^{n+2}*\\left|\\nabla{H}\\right|^{n-1}*\\nabla{H})}"
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