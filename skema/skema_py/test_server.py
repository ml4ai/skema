import json
import shutil
from tempfile import TemporaryDirectory, TemporaryFile
from pathlib import Path

from fastapi.testclient import TestClient

from skema.skema_py.server import app

client = TestClient(app)


def test_ping():
    response = client.get("/code2fn/ping")
    assert response.status_code == 200


def test_fn_supported_file_extensions():
    """Test case for /code2fn/fn-supported-file-extensions endpoint."""
    response = client.get("/code2fn/fn-supported-file-extensions")
    assert response.status_code == 200
    assert response.json() == [".py", ".f95", ".f"]


def test_fn_given_filepaths():
    """Test case for /code2fn/fn-given-filpaths endpoint with all optional fields included."""
    system = {
        "files": ["example1.py", "dir/example2.py"],
        "blobs": [
            "greet = lambda: print('howdy!')\ngreet()",
            "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1",  # Content of dir/example2.py
        ],
        "system_name": "example-system",
        "root_name": "example-system",
        "comments": {
            "files": {
                "example-system/dir/example2.py": {
                    "comments": [
                        {"contents": "Variable declaration", "line_number": 0},
                        {"contents": "Function definition", "line_number": 2},
                    ],
                    "docstrings": {"foo": ["Increment the input variable"]},
                }
            }
        },
    }
    response = client.post("/code2fn/fn-given-filepaths", json=system)

    assert response.status_code == 200
    assert "modules" in response.json()


def test_fn_given_filepaths_optional_fields():
    """Test case for /code2fn/fn-given-filpaths endpoint with all optional fields excluded."""
    system = {
        "files": ["example1.py", "dir/example2.py"],
        "blobs": [
            "greet = lambda: print('howdy!')\ngreet()",
            "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1",  # Content of dir/example2.py
        ],
    }
    response = client.post("/code2fn/fn-given-filepaths", json=system)

    assert response.status_code == 200
    assert "modules" in response.json()


def test_fn_given_filepaths_zip():
    """Test case for /code2fn/fn-given-filpaths-zip endpoint."""
    system = {
        "files": ["example1.py", "dir/example2.py"],
        "blobs": [
            "greet = lambda: print('howdy!')\ngreet()",
            "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1",  # Content of dir/example2.py
        ],
        "system_name": "example-system",
        "root_name": "example-system",
    }
    with TemporaryDirectory() as tmp:
        system_path = Path(tmp) / system["root_name"]
        system_path.mkdir()
        system_zip_path = Path(tmp) / f"{system['root_name']}.zip"

        for index, file in enumerate(system["files"]):
            file_path = Path(tmp, system["root_name"], file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(system["blobs"][index])

        input_path = Path(tmp, system["root_name"])
        output_path = Path(tmp, f"{system['root_name']}.zip")
        shutil.make_archive(input_path, "zip", input_path)

        response = client.post(
            "/code2fn/fn-given-filepaths-zip",
            files={"zip_file": open(output_path, "rb")},
        )
        assert response.status_code == 200
        assert "modules" in response.json()


# TODO: Add more complex test case to test_get_pyacset
def test_get_pyacset():
    """Test case for /code2fn/get_pyacset endpoint."""
    ports = {
        "opis": ["opi1", "opi2"],
        "opos": ["opo1", "opo2"],
    }
    response = client.post("/code2fn/get-pyacset", json=ports)
    assert response.status_code == 200
