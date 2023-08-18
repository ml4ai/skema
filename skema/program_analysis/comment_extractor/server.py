from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from typing import List, Union, Optional
from fastapi import FastAPI, File, UploadFile

import skema.program_analysis.comment_extractor.comment_extractor as comment_service
from skema.program_analysis.comment_extractor.model import (
    SingleFileCommentRequest,
    SingleFileCommentResponse,
    MultiFileCommentRequest,
    MultiFileCommentResponse,
    SupportedLanguageResponse,
)

SUPPORTED_FILE_EXTENSIONS = [".py", ".F", ".f", ".f95", ".for", ".m", ".r"]
EXTENSION_TO_LANGUAGE = {
    ".c": "c",
    ".cpp": "cpp",
    ".py": "python",
    ".F": "fortran",
    ".f": "fortran",
    ".f95": "fortran",
    ".for": "fortran",
    ".m": "matlab",
    ".r": "r",
}

app = FastAPI()


@app.get("/comments-get-supported-languages", summary="")
async def comments_get_supported_languages() -> SupportedLanguageResponse:
    """Endpoint for checking which type of comments are supported for each language."""
    return comment_service.get_supported_languages()


@app.get("/comments-get-supported-file-extensions")
async def comments_get_supported_file_extensions() -> List[str]:
    "Endpoint for checking which file extensions are supported for comment extraction."
    return SUPPORTED_FILE_EXTENSIONS


@app.post("/comments-extract", summary="")
async def comments_extract(
    request: SingleFileCommentRequest,
) -> SingleFileCommentResponse:
    """Endpoing for extracting comments from a single file.

    ### Python example
    '''
    request = {
        "source": "#Single line Python comment",
        "language": "python"
    }
    response = requests.post(URL, json=request)
    """
    return comment_service.extract_comments_single(request)


@app.post("/comments-extract-zip", summary="")
async def comments_extract_zip(zip_file: UploadFile = File()) -> MultiFileCommentResponse:
    """
    Endpoint for extracting comment from a zip archive of arbitrary depth and structure.
    All source files with a supported file extension will be processed as a single GrometFNModuleCollection.

    ### Python example
    ```
    files = {"zip_file": open("path/to/zip.zip")}
    requests.post(URL, files=files)
    """
    request = {"files": {}}
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            file_suffix = file_obj.suffix

            if file_suffix in SUPPORTED_FILE_EXTENSIONS:
                request["files"][file] = {
                    "language": EXTENSION_TO_LANGUAGE[file_suffix],
                    "source": zip.open(file).read(),
                }
                
    return comment_service.extract_comments_multi(
        MultiFileCommentRequest.parse_obj(request)
    )
