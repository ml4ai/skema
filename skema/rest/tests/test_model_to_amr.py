import asyncio
import requests
from zipfile import ZipFile
from io import BytesIO

from fastapi import UploadFile

from skema.rest.workflows import llm_assisted_codebase_to_pn_amr

CHIME_SIR_URL = (
    "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/CHIME-SIR-model.zip"
)


def test_any_amr_chime_sir():
    """Unit test for checking that Chime-SIR model produces any AMR"""
    response = requests.get(CHIME_SIR_URL)
    zip = BytesIO(response.content)

    upload_file = UploadFile(file=zip, filename="chime_sir.zip")
    amr = asyncio.run(llm_assisted_codebase_to_pn_amr(upload_file))

    assert "model" in amr
