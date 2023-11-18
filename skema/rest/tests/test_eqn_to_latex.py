from pathlib import Path
from fastapi.testclient import TestClient
from skema.rest.workflows import app

client = TestClient(app)


def test_post_image_to_latex():
    '''Test case for /images/equations-to-latex endpoint.'''

    cwd = Path(__file__).parents[0]
    image_path = cwd / "data" / "img2latex" / "halfar.png"
    files = {
        "data": open(image_path, "rb"),
    }
    
    endpoint = "/images/equations-to-latex"
    response = client.post(endpoint, files=files)
    expected = '"\\frac{d H}{dt}=\\nabla \\cdot {(\\Gamma*H^{n+2}*\\left|\\nabla{H}\\right|^{n-1}*\\nabla{H})}"'
    # check for route's existence
    assert any(route.path == endpoint for route in app.routes) == True, "{endpoint} does not exist for app"
    # check status code
    assert response.status_code == 200, f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response
    assert response.text == expected, f"Response should be {expected}, but instead received {response.text}"
