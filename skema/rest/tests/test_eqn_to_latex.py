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

    response = client.post("/images/equations-to-latex", files=files)
    expected = '"\\frac{d H}{dt}=\\nabla \\cdot {(\\Gamma*H^{n+2}*\\left|\\nabla{H}\\right|^{n-1}*\\nabla{H})}"'

    assert response.status_code == 200, "Request was unsuccessful"
    assert response.text == expected, f"Response should be {expected}, but instead received {response.text}"
