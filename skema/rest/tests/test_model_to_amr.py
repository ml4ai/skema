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
import time

CHIME_SIR_URL = (
    "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/CHIME-SIR-model.zip"
)

def test_any_amr_chime_sir():
    """
    Unit test for checking that Chime-SIR model produces any AMR. This test zip contains 4 versions of CHIME SIR.
    This will test if just the core dynamics works, the whole script, and also rewritten scripts work. 
    """
    response = requests.get(CHIME_SIR_URL)
    zip_bytes = BytesIO(response.content)

    # NOTE: For CI we are unable to use the LLM assisted functions due to API keys
    # So, we will instead mock the output for those functions instead
    dyn1 = Dynamics(name="CHIME_SIR-old.py", description=None, block=["L21-L31"])
    dyn2 = Dynamics(name="CHIME_SIR.py", description=None, block=["L101-L121"])
    dyn3 = Dynamics(name="CHIME_SIR_core.py", description=None, block=["L1-L9"])
    dyn4 = Dynamics(name="CHIME_SIR_while_loop.py", description=None, block=["L161-L201"])
    llm_mock_output = [dyn1, dyn2, dyn3, dyn4]

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
    logging = []
    for i in range(len(blobs)):
        if line_begin[i] == line_end[i]:
            print("failed linespan")
        else:
            blobs[i] = "".join(blobs[i].splitlines(keepends=True)[line_begin[i]:line_end[i]])
            try:
                time.sleep(0.5)
                code_snippet_response = asyncio.run(
                        code_snippets_to_pn_amr(
                            System(
                                files=[files[i]],
                                blobs=[blobs[i]],
                            )
                        )
                    )
                if "model" in code_snippet_response:
                    amrs.append(code_snippet_response)
                else:
                    print("snippets failure")
                    logging.append(f"{files[i]} failed to parse an AMR from the dynamics")
            except:
                print("except hit")
                logging.append(f"{files[i]} failed to parse an AMR from the dynamics")
    # we will return the amr with most states, in assumption it is the most "correct"
    # by default it returns the first entry
    print(f"amrs: {amrs}\n")
    amr = amrs[0]
    print(f"initial amr: {amr}\n")
    for temp_amr in amrs:
        try:
            temp_len = len(temp_amr["model"]["states"])
            amr_len = len(amr["model"]["states"])
            if temp_len > amr_len:
                amr = temp_amr
        except:
            continue
    print(f"final amr: {amr}\n")
    # For this test, we are just checking that AMR was generated without crashing. We are not checking for accuracy.
    assert "model" in amr, f"'model' should be in AMR response, but got {amr}"

