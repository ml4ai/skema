import os
import tempfile
from pathlib import Path
from typing import List, Optional
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from fastapi import FastAPI, Body, File, UploadFile
from pydantic import BaseModel

import skema.skema_py.acsets
import skema.skema_py.petris

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.snippet_ingester import process_snippet
from skema.utils.fold import dictionary_to_gromet_json, del_nulls


class Ports(BaseModel):
    opis: List[str]
    opos: List[str]


class System(BaseModel):
    files: List[str] = []
    blobs: List[str]
    system_name: Optional[str] = ""
    root_name: Optional[str] = ""


def system_to_gromet(system: System):
    """Convert a System to Gromet JSON"""

    # The CODE2FN Pipeline requires a file path as input.
    # We are receiving a serialized version of the code system as input, so we must store the file in a temporary directory.
    # This temp directory only persists during execution of the CODE2FN pipeline.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create files and intermediate directories
        for index, file in enumerate(system.files):
            file_path = Path(
                tmp_path, system.root_name, file
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(system.blobs[index])

        # Create system_filepaths.txt
        system_filepaths = Path(
            tmp_path, "system_filepaths.txt"
        )
        system_filepaths.write_text("\n".join(system.files))

        ## Run pipeline
        gromet_collection = process_file_system(
            system.system_name,
            str(Path(tmp_path, system.root_name)),
            str(system_filepaths),
        )

    # Convert gromet data-model to json
    gromet_collection_dict = gromet_collection.to_dict()
    return dictionary_to_gromet_json(del_nulls(gromet_collection_dict))


app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The skema-py service is running."


@app.post(
    "/fn-given-filepaths",
    summary=(
        "Send a system of code and filepaths of interest,"
        " get a GroMEt FN Module collection back."
    ),
)
async def fn_given_filepaths(system: System):
    """
    Endpoint for generating Gromet JSON from a .

    ### Python example
    ```
    import requests

    system = {
      "files": ["exp1.py"],
      "blobs": ["x=2"]
    }
    response = requests.post("http://0.0.0.0:8000/fn-given-filepaths", json=system)
    gromet_json = response.json()
    """
    return system_to_gromet(system)


@app.post(
    "/fn-given-filepaths-zip",
    summary=(
        "Send a zip file containing a code system,"
        " get a GroMEt FN Module collection back."
    ),
)
async def root(zip_file: UploadFile = File()):
    """
    Endpoint for generating Gromet JSON from a zip archive.

    ### Python example
    ```
    import requests

    files = {
      "zip_file": open("code_system.zip", "rb"),
    }
    response = requests.post("http://0.0.0.0:8000/fn-given-filepaths-zip", files=files)
    gromet_json = response.json()
    """

    # Currently, we rely on the file extension to know which language front end to send the source code to.
    supported_file_extensions = [".py", ".f", ".f95"]

    # To process a zip file, we first convert it to a System object, and then pass it to system_to_gromet.
    files = []
    blobs = []
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in supported_file_extensions:
                files.append(file)
                blobs.append(zip.open(file).read())

    zip_obj = Path(zip_file.filename)
    system_name = zip_obj.name
    root_name = zip_obj.name

    system = System(
        files=files, blobs=blobs, system_name=system_name, root_name=root_name
    )
    return system_to_gromet(system)


@app.post(
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
