import json
import os
import tempfile
import subprocess
from shutil import which
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from fastapi import APIRouter, FastAPI, Body, File, UploadFile
from pydantic import BaseModel, Field

import skema.skema_py.acsets
import skema.skema_py.petris

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.snippet_ingester import process_snippet
from skema.program_analysis.fn_unifier import align_full_system
from skema.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
from skema.program_analysis.comments import (
    CodeComments,
    SingleFileCodeComments,
    MultiFileCodeComments,
)

FN_SUPPORTED_FILE_EXTENSIONS = [".py", ".f95", ".f"]


class Ports(BaseModel):
    opis: List[str]
    opos: List[str]


class System(BaseModel):
    files: List[str] = Field(
        description="The relative file path from the directory specified by `root_name`, corresponding to each entry in `blobs`",
        example=["example1.py", "dir/example2.py"],
    )
    blobs: List[str] = Field(
        description="Contents of each file to be analyzed",
        example=[
            "greet = lambda: print('howdy!')\ngreet()",
            "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1",
        ],
    )
    system_name: Optional[str] = Field(
        default=None,
        description="A model name to associate with the provided code",
        example="example-system",
    )
    root_name: Optional[str] = Field(
        default=None,
        description="The name of the code system's root directory.",
        example="example-system",
    )
    comments: Optional[CodeComments] = Field(
        default=None,
        description="A CodeComments object representing the comments extracted from the source code in 'blobs'. Can provide comments for a single file (SingleFileCodeComments) or multiple files (MultiFileCodeComments)",
        example={
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
    )


def system_to_enriched_system(system: System) -> System:
    """Takes a System as input and enriches it with comments by running the Rust comment extractor."""

    # The skema_rs_path is the path to the Rust comment extractor.
    # It is used to set the cwd when running subprocess to simplify the execution process.
    skema_rs_path = Path(__file__).parent.parent / "skema-rs" / "comment_extraction"

    # Check if Rust is installed on the system. If not, then return the system as is.
    if not which("cargo"):
        print("HERE")
        return system

    # Create a temporary directory to store the input source files.
    # The Rust comment_extractor requires that the input exists on disk.
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    
    comments = {"files": {}}
    for file, blob in zip(system.files, system.blobs):
        file_path = Path(tmp_path, system.root_name or "", file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(blob)

        result = subprocess.run(
            ["cargo", "run", "--", str(file_path)],
            cwd=skema_rs_path,
            stdout=subprocess.PIPE,
        )
        comment_path = Path(system.root_name or "") / file
        comments["files"][str(comment_path)] = json.loads(
            str(result.stdout, encoding="UTF-8")
        )
    # Build the MultiFileCodeComments model from the comments dict
    system.comments = MultiFileCodeComments.model_validate(comments)

    # Cleanup the temporary directory
    tmp.cleanup()

    return system


def system_to_gromet(system: System):
    """Convert a System to Gromet JSON"""

    # The CODE2FN Pipeline requires a file path as input.
    # We are receiving a serialized version of the code system as input, so we must store the file in a temporary directory.
    # This temp directory only persists during execution of the CODE2FN pipeline.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create files and intermediate directories
        for index, file in enumerate(system.files):
            file_path = Path(tmp_path, system.root_name or "", file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(system.blobs[index])

        # Create system_filepaths.txt
        system_filepaths = Path(tmp_path, "system_filepaths.txt")
        system_filepaths.write_text("\n".join(system.files))

        ## Run pipeline
        gromet_collection = process_file_system(
            system.system_name or "",
            str(Path(tmp_path, system.root_name or "")),
            str(system_filepaths),
        )

    # Attempt to enrich the system with comments. May return the same system if Rust isn't insalled.
    if not system.comments:
        system = system_to_enriched_system(system)

    # If comments are included in request or added in the enriching process, run the unifier to add them to the Gromet
    if system.comments:
        align_full_system(gromet_collection, system.comments)

    # Explicitly call to_dict on any metadata object
    # NOTE: Only required because of fault in swagger-codegen
    for i, module in enumerate(gromet_collection.modules):
        for j, metadata_list in enumerate(module.metadata_collection):
            for k, metadata in enumerate(metadata_list):
                gromet_collection.modules[i].metadata_collection[j][
                    k
                ] = metadata.to_dict()

    # Convert Gromet data-model to dict for return
    return gromet_collection.to_dict()


router = APIRouter()


@router.get("/ping", summary="Ping endpoint to test health of service")
def ping() -> int:
    return 200


@router.get(
    "/fn-supported-file-extensions",
    summary="Endpoint for checking which files extensions are currently supported by code2fn pipeline.",
    response_model=List[str],
)
def fn_supported_file_extensions():
    """
    Returns a List[str] where each entry in the list represents a file extension.

    ### Python example
    ```
    import requests

    response = requests.get("http://0.0.0.0:8000/fn-supported-file-extensions")
    supported_extensions = response.json()

    """
    return FN_SUPPORTED_FILE_EXTENSIONS


@router.post(
    "/fn-given-filepaths",
    summary=(
        "Send a system of code and filepaths of interest,"
        " get a GroMEt FN Module collection back."
    ),
)
async def fn_given_filepaths(system: System):
    """
    Endpoint for generating Gromet JSON from a serialized code system.
    ### Python example

    ```
    import requests

    # Single file
    system = {
      "files": ["exp1.py"],
      "blobs": ["x=2"]
    }
    response = requests.post("http://0.0.0.0:8000/fn-given-filepaths", json=system)
    gromet_json = response.json()

    # Multi file
    system = {
      "files": ["exp1.py", "exp1.f"],
      "blobs": ["x=2", "program exp1\\ninteger::x=2\\nend program exp1"],
      "system_name": "exp1",
      "root_name": "exp1"
    }
    response = requests.post("http://0.0.0.0:8000/fn-given-filepaths", json=system)
    gromet_json = response.json()
    """

    return system_to_gromet(system)


@router.post(
    "/fn-given-filepaths-zip",
    summary=(
        "Send a zip file containing a code system,"
        " get a GroMEt FN Module collection back."
    ),
)
async def fn_given_filepaths_zip(zip_file: UploadFile = File()):
    """
    Endpoint for generating Gromet JSON from a zip archive of arbitrary depth and structure.
    All source files with a supported file extension (/fn-supported-file-extensions) will be processed as a single GrometFNModuleCollection.

    ### Python example
    ```
    import requests
    import shutil
    from pathlib import Path

    # Format input/output paths
    input_name = "system_test"
    output_name = "system_test.zip"
    input_path = Path("/data") / "skema" / "code" / input_name
    output_path = Path("/data") / "skema" / "code" / output_name

    # Convert source directory to zip archive
    shutil.make_archive(input_path, "zip", input_path)

    files = {
      "zip_file": open(output_path, "rb"),
    }
    response = requests.post("http://0.0.0.0:8000/fn-given-filepaths-zip", files=files)
    gromet_json = response.json()
    """

    # To process a zip file, we first convert it to a System object, and then pass it to system_to_gromet.
    files = []
    blobs = []
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in FN_SUPPORTED_FILE_EXTENSIONS:
                files.append(file)
                blobs.append(zip.open(file).read())

    zip_obj = Path(zip_file.filename)
    system_name = zip_obj.stem
    root_name = zip_obj.stem

    system = System(
        files=files, blobs=blobs, system_name=system_name, root_name=root_name
    )

    return system_to_gromet(system)


@router.post(
    "/get-pyacset",
    summary=("Get PyACSet for a given model"),
)
async def get_pyacset(ports: Ports):
    opis, opos = ports.opis, ports.opos
    petri = skema.skema_py.petris.Petri()
    petri.add_species(len(opos))
    trans = skema.skema_py.petris.Transition
    petri.add_parts(trans, len(opis))

    for i, tran in enumerate(opis):
        petri.set_subpart(i, skema.skema_py.petris.attr_tname, opis[i])

    for j, spec in enumerate(opos):
        petri.set_subpart(j, skema.skema_py.petris.attr_sname, opos[j])

    return petri.write_json()


app = FastAPI()
app.include_router(
    router,
    prefix="/code2fn",
    tags=["code2fn"],
)
