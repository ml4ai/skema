import logging
import os
import tempfile
from typing import List

from fastapi import FastAPI, Body
from pydantic import BaseModel

import skema.skema_py.acsets
import skema.skema_py.petris

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.utils.fold import dictionary_to_gromet_json, del_nulls


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/ping") == -1


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


class Ports(BaseModel):
    opis: List[str]
    opos: List[str]


class System(BaseModel):
    files: List[str]
    blobs: List[str]
    system_name: str
    root_name: str


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
    # Create a tempory directory to store module
    with tempfile.TemporaryDirectory() as tmp:
        # Recreate module structure
        for index, file in enumerate(system.files):
            full_path = os.path.join(tmp, system.root_name, file)
            # Create file and intermediate directories first
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(system.blobs[index])

        # Create system_filepaths.txt file
        system_filepaths = os.path.join(tmp, "system_filepaths.txt")
        with open(system_filepaths, "w") as f:
            f.writelines(file + "\n" for file in system.files)

        ## Run pipeline
        gromet_collection = process_file_system(
            system.system_name,
            os.path.join(tmp, system.root_name),
            system_filepaths,
        )

    # Convert output to json
    gromet_collection_dict = gromet_collection.to_dict()
    return dictionary_to_gromet_json(del_nulls(gromet_collection_dict))


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
