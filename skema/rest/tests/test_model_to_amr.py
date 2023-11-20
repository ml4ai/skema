import asyncio
import requests
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from fastapi import UploadFile

from skema.rest.workflows import (
    llm_assisted_codebase_to_pn_amr,
    code_snippets_to_pn_amr,
)
from skema.rest.llm_proxy import Dynamics
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.skema_py.server import System

CHIME_SIR_URL = (
    "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/CHIME-SIR-model.zip"
)


def test_any_amr_chime_sir():
    """Unit test for checking that Chime-SIR model produces any AMR"""
    response = requests.get(CHIME_SIR_URL)
    zip_bytes = BytesIO(response.content)

    # NOTE: For CI we are unable to use the LLM assisted functions due to API keys
    # So, we will instead mock the output for those functions instead
    llm_mock_output = Dynamics(name=None, description=None, block=["L21-L31"])
    lines = llm_mock_output.block[0].split("-")
    line_begin = max(int(lines[0][1:]) - 1, 0)
    line_end = int(lines[1][1:])

    # This is a more succinct verion of the slicing code in llm_assisted_codebase_to_pn_amr
    with ZipFile(zip_bytes) as zip, zip.open(zip.namelist()[0]) as first_file:
        files = [zip.namelist()[0]]
        blobs = [
            "".join(
                first_file.read()
                .decode("utf-8")
                .splitlines(keepends=True)[line_begin:line_end]
            )
        ]
    print(f"SKEMA_RS_ADDESS:\t{SKEMA_RS_ADDESS}")

    amr = asyncio.run(
        code_snippets_to_pn_amr(
            System(
                files=files,
                blobs=blobs,
            )
        )
    )
    # For this test, we are just checking that AMR was generated without crashing. We are not checking for accuracy.
    assert "model" in amr, f"'model' should be in AMR response, but got {amr}"

