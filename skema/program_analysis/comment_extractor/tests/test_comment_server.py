import shutil
from tempfile import TemporaryDirectory, TemporaryFile
from pathlib import Path
from typing import List
from fastapi.testclient import TestClient

import skema.program_analysis.comment_extractor.comment_extractor as comment_service
from skema.program_analysis.comment_extractor.server import app

client = TestClient(app)

def test_comments_get_supported_languages():
    '''Test case for /comments-get-supported-languages endpoint.'''
    response = client.get("/comment_service/comments-get-supported-languages")
    assert response.status_code == 200
    
    languages = comment_service.SupportedLanguageResponse.parse_obj(response.json())
    assert isinstance(languages, comment_service.SupportedLanguageResponse)
    assert len(languages.languages) > 0


def test_comments_get_supprted_file_extensions():
    '''Test cast for /comments-get-supported-file-extensions'''
    response = client.get("/comment_service/comments-get-supported-file-extensions")
    assert response.status_code == 200
    
    extensions = response.json()
    assert isinstance(extensions, List)
    assert len(extensions) > 0

def test_comments_extract():
    '''Test cast for /comments-extract endpoint'''
    request = {
        "source": "# Simple comment extraction example",
        "language": "python"
    }
    response = client.post("/comment_service/comments-extract", json=request)
    assert response.status_code == 200

    comments = comment_service.SingleFileCommentResponse.parse_obj(response.json())
    assert isinstance(comments, comment_service.SingleFileCommentResponse)
    

def test_comments_extract_zip():
    """Test case for /comments-extract-zip endpoint"""
    system = {
        "files": ["example1.py", "dir/example2.py"],
        "blobs": [
            "greet = lambda: print('howdy!')\ngreet()",
            "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1",  # Content of dir/example2.py
        ],
        "root_name": "example",
        "system_name": "example"
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
            "/comment_service/comments-extract-zip",
            files={"zip_file": open(output_path, "rb")},
        )
        assert response.status_code == 200
        
        comments = comment_service.MultiFileCommentResponse.parse_obj(response.json())
        assert isinstance(comments, comment_service.MultiFileCommentResponse)