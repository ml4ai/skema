from pathlib import Path

from fastapi.testclient import TestClient

import skema.program_analysis.comment_extractor.comment_extractor as comment_service
from skema.program_analysis.comment_extractor.server import app

EXAMPLES_PATH = Path(__file__).parent / "examples"

client = TestClient(app)

def test_extract_c():
    c_source_path = EXAMPLES_PATH / "comments.c"
    request = {
        "language": "c",
        "source": c_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200


def test_extract_cpp():
    cpp_source_path = EXAMPLES_PATH / "comments.cpp"
    request = {
        "language": "cpp",
        "source": cpp_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200

def test_extract_fortran():
    fortran_source_path = EXAMPLES_PATH / "comments.f95"
    request = {
        "language": "fortran",
        "source": fortran_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200

def test_extract_matlab():
    matlab_source_path = EXAMPLES_PATH / "comments.m"
    request = {
        "language": "matlab",
        "source": matlab_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200

def test_extract_python():
    python_source_path = EXAMPLES_PATH / "comments.py"
    request = {
        "language": "python",
        "source": python_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200

def test_extract_r():
    r_source_path = EXAMPLES_PATH / "comments.r"
    request = {
        "language": "r",
        "source": r_source_path.read_text()
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200
