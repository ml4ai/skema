import os
import tempfile
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.utils.fold import dictionary_to_gromet_json, del_nulls


class System(BaseModel):
    files: List[str]
    blobs: List[str]
    system_name: str
    root_name: str


app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The Code2FN service is running."


@app.post(
    "/fn-given-filepaths",
    summary=(
        "Send a system of code and filepaths of interest,"
        " get a GroMEt FN Module collection back."
    ),
)
async def root(system: System):
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
