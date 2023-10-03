import json
import shutil
from tempfile import TemporaryDirectory, TemporaryFile
from pathlib import Path

from fastapi.testclient import TestClient

from skema.skema_py.server import app
from skema.gromet.metadata.debug import Debug

client = TestClient(app)


def test_ping():
    '''Test case for /code2fn/ping endpoint.'''
    response = client.get("/code2fn/ping")
    assert response.status_code == 200


def test_fn_supported_file_extensions():
    """Test case for /code2fn/fn-supported-file-extensions endpoint."""
    response = client.get("/code2fn/fn-supported-file-extensions")
    assert response.status_code == 200
    assert len(response.json()) > 0


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
                    "single": [
                        {"content": "Variable declaration", "line_number": 0},
                        {"content": "Function definition", "line_number": 2},
                    ],
                    "multi": [],
                    "docstring": [
                        {"function_name": "foo", "content": ["Increment the input variable"], "start_line_number": 4, "end_line_number": 4}
                    ]
                }
            }
        },
    }
    response = client.post("/code2fn/fn-given-filepaths", json=system)
    print(response.json())
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


def test_no_supported_files():
    system = {
        "files": ["unsupported1.git", "unsupported2.lock"],
        "blobs": [
            "This is not a source code file.",
            "This is not a source code file.",  
        ],
        "system_name": "unsupported-system",
        "root_name": "unsupported-system",
    }

    response = client.post("/code2fn/fn-given-filepaths", json=system)
    assert response.status_code == 200

    gromet_collection = response.json()
    assert "metadata_collection" in gromet_collection
    assert len(gromet_collection["metadata_collection"]) == 1 # Only one element (GrometFNModuleCollection) should create metadata in this metadata_collection
    assert len(gromet_collection["metadata_collection"][0]) == 1 # There should only be one ERROR Debug metadata since there are no source files to process.
    assert gromet_collection["metadata_collection"][0][0]["gromet_type"] == "Debug"
    assert gromet_collection["metadata_collection"][0][0]["severity"] == "ERROR"

def test_partial_supported_files():
    system = {
        "files": ["supported.py", "unsupported.lock"],
        "blobs": [
            "x=2",
            "This is not a source code file.",  
        ],
        "system_name": " mixed-system",
        "root_name": "mixed-system",
    }
    
    response = client.post("/code2fn/fn-given-filepaths", json=system)
    assert response.status_code == 200

    gromet_collection = response.json()
    assert "metadata_collection" in gromet_collection
    assert len(gromet_collection["metadata_collection"]) == 1 # Only one element (GrometFNModuleCollection) should create metadata in this metadata_collection
    assert len(gromet_collection["metadata_collection"][0]) == 1 # There should only be one WARNING Debug metadata since is a single unsupported file.
    assert gromet_collection["metadata_collection"][0][0]["gromet_type"] == "Debug"
    assert gromet_collection["metadata_collection"][0][0]["severity"] == "WARNING"

# TODO: Add more complex test case to test_get_pyacset
def test_get_pyacset():
    """Test case for /code2fn/get_pyacset endpoint."""
    ports = {
        "opis": ["opi1", "opi2"],
        "opos": ["opo1", "opo2"],
    }
    response = client.post("/code2fn/get-pyacset", json=ports)
    assert response.status_code == 200
