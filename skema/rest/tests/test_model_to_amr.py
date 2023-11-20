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
SIMPLE_SIR_URL = (
    "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/code_sir.zip"
)

def test_any_amr_chime_sir():
    """Unit test for checking that Chime-SIR model produces any AMR"""
    response = requests.get(CHIME_SIR_URL)
    zip_bytes = BytesIO(response.content)

    # NOTE: For CI we are unable to use the LLM assisted functions due to API keys
    # So, we will instead mock the output for those functions instead
    llm_mock_output = Dynamics(name=None, description=None, block=["L21-L31"])

    line_begin = []
    line_end = []
    files = []
    blobs = []
    amrs = []
    for linespan in llm_mock_output:
        lines = linespan.block[0].split("-")
        line_begin.append(
            max(int(lines[0][1:]) - 1, 0)
        )  # Normalizing the 1-index response from llm_proxy
        line_end.append(int(lines[1][1:]))

        # So we are required to do the same when slicing the source code using its output.
    with ZipFile(zip_bytes, "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in [".py"]:
                files.append(file)
                blobs.append(zip.open(file).read().decode("utf-8"))

    # The source code is a string, so to slice using the line spans, we must first convert it to a list.
    # Then we can convert it back to a string using .join
    for i in range(len(blobs)):
        if line_begin[i] == line_end[i]:
            print("failed linespan")
        else:
            blobs[i] = "".join(blobs[i].splitlines(keepends=True)[line_begin[i]:line_end[i]])
            amrs.append(
                asyncio.run(
                    code_snippets_to_pn_amr(
                        System(
                            files=files,
                            blobs=blobs,
                        )
                    )
                )
            )
    # we will return the amr with most states, in assumption it is the most "correct"
    # by default it returns the first entry
    amr = amrs[0]
    for temp_amr in amrs:
        try:
            temp_len = len(temp_amr["model"]["states"])
            amr_len = len(amr["model"]["states"])
            if temp_len > amr_len:
                amr = temp_amr
        except:
            continue

    # For this test, we are just checking that AMR was generated without crashing. We are not checking for accuracy.
    assert "model" in amr, f"'model' should be in AMR response, but got {amr}"

def test_any_amr_simple_sir():
    """Unit test for checking that Simple-SIR model zip produces any AMR"""
    response = requests.get(SIMPLE_SIR_URL)
    zip_bytes = BytesIO(response.content)

    # NOTE: For CI we are unable to use the LLM assisted functions due to API keys
    # So, we will instead mock the output for those functions instead
    llm_mock_output = [Dynamics(name="code/code.py", description=None, block=["L2-L10"])]

    line_begin = []
    line_end = []
    files = []
    blobs = []
    amrs = []
    for linespan in llm_mock_output:
        lines = linespan.block[0].split("-")
        line_begin.append(
            max(int(lines[0][1:]) - 1, 0)
        )  # Normalizing the 1-index response from llm_proxy
        line_end.append(int(lines[1][1:]))

        # So we are required to do the same when slicing the source code using its output.
    with ZipFile(zip_bytes, "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in [".py"]:
                files.append(file)
                blobs.append(zip.open(file).read().decode("utf-8"))

    # The source code is a string, so to slice using the line spans, we must first convert it to a list.
    # Then we can convert it back to a string using .join
    for i in range(len(blobs)):
        if line_begin[i] == line_end[i]:
            print("failed linespan")
        else:
            blobs[i] = "".join(blobs[i].splitlines(keepends=True)[line_begin[i]:line_end[i]])
            amrs.append(
                asyncio.run(
                    code_snippets_to_pn_amr(
                        System(
                            files=files,
                            blobs=blobs,
                        )
                    )
                )
            )
    # we will return the amr with most states, in assumption it is the most "correct"
    # by default it returns the first entry
    amr = amrs[0]
    for temp_amr in amrs:
        try:
            temp_len = len(temp_amr["model"]["states"])
            amr_len = len(amr["model"]["states"])
            if temp_len > amr_len:
                amr = temp_amr
        except:
            continue

    # For this test, we are just checking that AMR was generated without crashing. We are not checking for accuracy.
    assert "model" in amr, f"'model' should be in AMR response, but got {amr}"
