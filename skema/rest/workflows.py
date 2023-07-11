# -*- coding: utf-8 -*-
"""
End-to-end skema workflows
"""


from typing import List
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.rest import schema, utils, comments_proxy
from skema.program_analysis.comments import MultiFileCodeComments
from skema.img2mml import eqn2mml
from skema.skema_py import server as code2fn
from fastapi import APIRouter, File, UploadFile
from starlette.responses import JSONResponse
import requests
import json


router = APIRouter()


# equation images -> mml -> amr
@router.post("/images/base64/equations-to-amr", summary="Equations (base64 images) → MML → AMR")
async def equations_to_amr(data: schema.EquationImagesToAMR):
    """
    Converts images of equations to AMR.

    ### Python example
    ```
    from pathlib import Path
    import base64
    import requests


    with Path("bayes-rule-white-bg.png").open("rb") as infile:
      img_bytes = infile.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    r = requests.post(url, data=img_b64)
    print(r.text)

    images_bytes = [open("eq1.png", "rb").read(), open("eq2.png", "rb").read()]

    images_b64 = [base64.b64encode(img_bytes).decode("utf-8") for img_bytes in images_bytes]

    url = "0.0.0.0"
    r = requests.post(f"{url}/workflows/images/base64/equations-to-amr", json={"images": images_b64, "model": "regnet"})
    r.json()
    """
    mml: List[str] = [eqn2mml.b64_image_to_mml(img) for img in data.images]
    payload = {"mathml": mml, "model": data.model}
    # FIXME: why is this a PUT?
    res = requests.put(
        f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload
    )
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE PUT /mathml/amr failed to process payload", "payload": payload})
    return res.json()


# tex equations -> pmml -> amr
@router.post("/latex/equations-to-amr", summary="Equations (LaTeX) → pMML → AMR")
async def equations_to_amr(data: schema.EquationLatexToAMR):
    """
    Converts equations (in LaTeX) to AMR.

    ### Python example
    ```
    import requests

    equations = [
      "\\frac{\\delta x}{\\delta t} = {\\alpha x} - {\\beta x y}",
      "\\frac{\\delta y}{\\delta t} = {\\alpha x y} - {\\gamma y}"
    ]
    url = "0.0.0.0"
    r = requests.post(f"{url}/workflows/latex/equations-to-amr", json={"equations": equations, "model": "regnet"})
    r.json()
    """
    mml: List[str] = [
        utils.clean_mml(eqn2mml.get_mathml_from_latex(tex))
        for tex in data.equations
    ]
    payload = {"mathml": mml, "model": data.model}
    res = requests.put(
        f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload
    )
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE PUT /mathml/amr failed to process payload", "payload": payload})
    return res.json()


# code snippets -> fn -> petrinet amr
@router.post("/code/snippets-to-pn-amr", summary="Code snippets → PetriNet AMR")
async def code_snippets_to_pn_amr(system: code2fn.System):
    if system.comments == None:
        system = await code2fn.system_to_enriched_system(system)
    gromet = await code2fn.fn_given_filepaths(system)
    res = requests.post(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE POST /models/PN failed to process payload", "payload": gromet})
    return res.json()


# code snippets -> fn -> regnet amr
@router.post("/code/snippets-to-rn-amr", summary="Code snippets → RegNet AMR")
async def code_snippets_to_rn_amr(system: code2fn.System):
    if system.comments == None:
        system = await code2fn.system_to_enriched_system(system)
    gromet = await code2fn.fn_given_filepaths(system)
    res = requests.post(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE POST /models/RN failed to process payload", "payload": gromet})
    return res.json()

# zip archive -> fn -> petrinet amr
@router.post(
    "/code/codebase-to-pn-amr", summary="Code repo (zip archive) → PetriNet AMR"
)
async def repo_to_pn_amr(zip_file: UploadFile = File()):
    gromet = await code2fn.fn_given_filepaths_zip(zip_file)
    res = requests.post(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE POST /models/PN failed to process payload", "payload": gromet})
    return res.json()


# zip archive -> fn -> regnet amr
@router.post("/code/codebase-to-rn-amr", summary="Code repo (zip archive) → RegNet AMR")
async def repo_to_rn_amr(zip_file: UploadFile = File()):
    gromet = await code2fn.fn_given_filepaths_zip(zip_file)
    res = requests.post(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(status_code=400, content={"error": f"MORAE POST /models/RN failed to process payload", "payload": gromet})
    return res.json()
