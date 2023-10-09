import json
import yaml
import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from fastapi import APIRouter, FastAPI, Body, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import skema.skema_py.acsets
import skema.skema_py.petris

import skema.program_analysis.comment_extractor.server as comment_service
from skema.utils.fold import del_nulls
from skema.gromet.fn.gromet_fn_module_collection import GrometFNModuleCollection
from skema.gromet.metadata.debug import Debug
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.snippet_ingester import process_snippet
from skema.program_analysis.fn_unifier import align_full_system
from skema.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
from skema.program_analysis.comment_extractor.model import (
    SingleFileCommentRequest,
    SingleFileCommentResponse,
    MultiFileCommentRequest,
    MultiFileCommentResponse,
    CodeComments,
)
from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    LANGUAGES_YAML_FILEPATH,
)


def get_supported_languages() -> (List, Dict):
    """"""
    # We calculate the supported file extensions and mapping between extension and language by reading the languages.yaml file from tree_sitter_parsers
    languages_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())

    supported_file_extensions = []
    extension_to_language = {}
    for language, language_dict in languages_obj.items():
        if language_dict["supports_fn_extraction"]:
            supported_file_extensions.extend(language_dict["extensions"])
            extension_to_language.update(
                {extension: language for extension in language_dict["extensions"]}
            )

    return supported_file_extensions, extension_to_language


SUPPORTED_FILE_EXTENSIONS, EXTENSION_TO_LANGUAGE = get_supported_languages()


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
                    "single": [
                        {"content": "Variable declaration", "line_number": 0},
                        {"content": "Function definition", "line_number": 2},
                    ],
                    "multi": [],
                    "docstring": [
                        {
                            "content": ["Increment the input variable"],
                            "function_name": "foo",
                            "start_line_number": 5,
                            "end_line_number": 6,
                        }
                    ],
                }
            }
        },
    )


async def system_to_enriched_system(system: System) -> System:
    """Takes a System as input and enriches it with comments by running the tree-sitter comment extractor."""

    # Instead of making each proxy call seperatly, we will gather them
    coroutines = []
    file_paths = []
    for file, blob in zip(system.files, system.blobs):
        file_path = Path(system.root_name or "") / file
        if file_path.suffix not in SUPPORTED_FILE_EXTENSIONS:
            # Since we are enriching a system for unification, we only want to extract comments from source files we can also extract Gromet FN from.
            continue

        request = SingleFileCommentRequest(
            source=blob, language=EXTENSION_TO_LANGUAGE[file_path.suffix]
        )
        coroutines.append(comment_service.comments_extract(request))
        file_paths.append(file_path)
    results = await asyncio.gather(*coroutines)

    # Due to the nested structure of MultiFileCodeComments, it easier to work with a Dict.
    # Then, we can convert it using MutliFileCodeComments.model_validate()
    comments = {"files": {}}
    for file_path, result in zip(file_paths, results):
        comments["files"][str(file_path)] = result
    system.comments = MultiFileCommentResponse.parse_obj(comments)

    return system


async def system_to_gromet(system: System):
    """Convert a System to Gromet JSON"""

    # We maintain a log of warnings and error to pass back to the user.
    # This allows us to warn the user about unsupported file extensions.
    server_log = []

    # Check for unsupported files before processing. They will be removed and the user will be warned.
    unsupported_files = [
        file
        for file in system.files
        if Path(file).suffix not in SUPPORTED_FILE_EXTENSIONS
    ]
    for file in unsupported_files:
        unsupported_file_str = f"WARNING: Ingestion of file extension {Path(file).suffix} for file {file} is not supported and will be skipped."
        print(unsupported_file_str)
        system.files.remove(file)
        server_log.append(unsupported_file_str)

    # If there are no supported files, then we will return an empty GrometFNModuleCollection with a top-level Debug metadata
    if len(system.files) == 0:
        no_supported_file_str = "ERROR: The system does not contain any files with supported file extensions. All files will be skipped."
        print(no_supported_file_str)
        gromet_collection = GrometFNModuleCollection(
            metadata_collection=[[]], metadata=0
        )
        gromet_collection.metadata_collection[0].append(
            Debug(
                debug_type="code2fn", severity="ERROR", message=no_supported_file_str
            ).to_dict()  # There is a bug in Swagger that requires us to manually to_dict() this
        )
        return gromet_collection.to_dict()

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
        system = await system_to_enriched_system(system)

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

    # Add debug Metadata to Gromet objects
    if not gromet_collection.metadata_collection:
        gromet_collection.metadata_collection = [[]]
    for log in server_log:
        gromet_collection.metadata_collection[0].append(
            Debug(
                debug_type="code2fn", severity="WARNING", message=log
            ).to_dict()  # There is a bug in Swagger that requires us to manually to_dict() this
        )

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
    return SUPPORTED_FILE_EXTENSIONS


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

    return await system_to_gromet(system)


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
            if file_obj.suffix in SUPPORTED_FILE_EXTENSIONS:
                files.append(file)
                blobs.append(zip.open(file).read())

    zip_obj = Path(zip_file.filename)
    system_name = zip_obj.stem
    root_name = zip_obj.stem

    system = System(
        files=files, blobs=blobs, system_name=system_name, root_name=root_name
    )

    return await system_to_gromet(system)


@router.post(
    "/get-object-count",
    summary=("Endpoint for counting the number of boxes, wires, and ports in a Gromet object."),
)
async def get_object_count(gromet_object: Dict):
    """
    Endpoint for counting the number of boxes, wires, and ports in a Gromet object.
    ### Python example
    ```
    import requests

    system = {
        "files": ["example1.py"],
        "blobs": [
            "greet = lambda: print('howdy!')"
        ],
    }
    response = client.post("/code2fn/fn-given-filepaths", json=system)
    gromet_collection = response.json()
    response = client.post("/code2fn/get-object-count", json=gromet_collection)
    gromet_object_count = response.json()
    """

    gromet_keys = {
        "b": 0,
        "bf": 0,
        "opi": 0,
        "opo": 0,
        "pil": 0,
        "pol": 0,
        "wlopi": 0,
        "wll": 0,
        "wlf": 0,
        "wlc": 0,
        "wlopo": 0,
        "pof": 0,
        "pif": 0,
        "wfopi": 0,
        "wfl": 0,
        "wff": 0,
        "wfc": 0,
        "wfopo": 0,
        "pic": 0,
        "poc": 0,
        "wcopi": 0,
        "wcl": 0,
        "wcf": 0,
        "wcc": 0,
        "wcopo": 0,
    }

    def recurse(gromet_object: Dict):
        """Recursive walking function for Gromet"""
        for key, value in gromet_object.items():
            if key in gromet_keys:
                gromet_keys[key] += len(value)
            elif isinstance(value, List):
                for element in value:
                    if isinstance(element, Dict):
                        recurse(element)
            elif isinstance(value, Dict):
                recurse(value)

    # Its likely that the Gromet passed to this endpoint will not have None values removed.
    # So, we need to remove None values ahead of time.
    del_nulls(gromet_object)
    recurse(gromet_object)

    # We also will aggregate the boxes, wires, and ports to better support MORAE usecases
    gromet_keys["boxes"] = sum(
        [val for key, val in gromet_keys.items() if key.startswith("b")]
    )
    gromet_keys["wires"] = sum(
        [val for key, val in gromet_keys.items() if key.startswith("w")]
    )
    gromet_keys["ports"] = sum(
        [val for key, val in gromet_keys.items() if key.startswith(("p", "o"))]
    )

    return gromet_keys


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
