# -*- coding: utf-8 -*-
"""
End-to-end skema workflows
"""
import copy
import requests
from zipfile import ZipFile
from io import BytesIO
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from starlette.responses import JSONResponse

from skema.img2mml import eqn2mml
from skema.img2mml.eqn2mml import image2mathml_db
from skema.img2mml.api import get_mathml_from_bytes
from skema.rest import schema, utils, llm_proxy
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.skema_py import server as code2fn

router = APIRouter()


# equation images -> mml -> amr
@router.post(
    "/images/base64/equations-to-amr", summary="Equations (base64 images) → MML → AMR"
)
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
    mml: List[str] = [
        utils.clean_mml(eqn2mml.b64_image_to_mml(img)) for img in data.images
    ]
    payload = {"mathml": mml, "model": data.model}
    # FIXME: why is this a PUT?
    res = requests.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /mathml/amr failed to process payload with error {res.text}",
                "payload": payload,
            },
        )
    return res.json()


# equation images -> mml -> latex
@router.post("/images/equations-to-latex", summary="Equations (images) → MML → LaTeX")
async def equations_to_latex(data: UploadFile):
    """
    Converts images of equations to LaTeX.

    ### Python example
    ```
    Endpoint for generating LaTeX from an input image.

    ### Python example
    ```
    import requests

    files = {
      "data": open("bayes-rule-white-bg.png", "rb"),
    }
    r = requests.post("http://0.0.0.0:8000/images/equations-to-latex", files=files)
    print(r.text)
    """
    # Read image data
    image_bytes = await data.read()

    # pass image bytes to get_mathml_from_bytes function
    mml_res = get_mathml_from_bytes(image_bytes, image2mathml_db)
    response = requests.put(f"{SKEMA_RS_ADDESS}/mathml/latex", data=mml_res)
    # Check the response
    if response.status_code == 200:
        # The request was successful
        return response.text
    else:
        # The request failed
        print(f"Error: {response.status_code}")
        print(response.text)
        return f"Error: {response.status_code} {response.text}"


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
        utils.clean_mml(eqn2mml.get_mathml_from_latex(tex)) for tex in data.equations
    ]
    payload = {"mathml": mml, "model": data.model}
    res = requests.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /mathml/amr failed to process payload with error {res.text}",
                "payload": payload,
            },
        )
    return res.json()


# pmml -> amr
@router.post("/pmml/equations-to-amr", summary="Equations pMML → AMR")
async def equations_to_amr(data: schema.MmlToAMR):
    payload = {"mathml": data.equations, "model": data.model}
    res = requests.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /mathml/amr failed to process payload with error {res.text}",
                "payload": payload,
            },
        )
    return res.json()


# code snippets -> fn -> petrinet amr
@router.post("/code/snippets-to-pn-amr", summary="Code snippets → PetriNet AMR")
async def code_snippets_to_pn_amr(system: code2fn.System):
    gromet = await code2fn.fn_given_filepaths(system)
    gromet, logs = utils.fn_preprocessor(gromet)
    res = requests.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload",
                "payload": gromet,
            },
        )
    return res.json()


""" TODO: The regnet endpoints are currently outdated
# code snippets -> fn -> regnet amr
@router.post("/code/snippets-to-rn-amr", summary="Code snippets → RegNet AMR")
async def code_snippets_to_rn_amr(system: code2fn.System):
    gromet = await code2fn.fn_given_filepaths(system)
    res = requests.put(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet)
    if res.status_code != 200:
        print(res.status_code)
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE POST /models/RN failed to process payload",
                "payload": gromet,
            },
        )
    return res.json()
"""


# zip archive -> fn -> petrinet amr
@router.post(
    "/code/codebase-to-pn-amr", summary="Code repo (zip archive) → PetriNet AMR"
)
async def repo_to_pn_amr(zip_file: UploadFile = File()):
    gromet = await code2fn.fn_given_filepaths_zip(zip_file)
    gromet, logs = utils.fn_preprocessor(gromet)
    res = requests.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload",
                "payload": gromet,
            },
        )
    return res.json()


# zip archive -> linespan -> snippet -> petrinet amr
@router.post(
    "/code/llm-assisted-codebase-to-pn-amr",
    summary="Code repo (zip archive) → PetriNet AMR",
)
async def llm_assisted_codebase_to_pn_amr(zip_file: UploadFile = File()):
    """Codebase->AMR workflow using an llm to extract the dynamics line span.
    ### Python example
    ```
    import requests

    files = {
        'zip_archive': open('model_source.zip')
    }
    response = requests.post("localhost:8000/workflows/code/llm-assisted-codebase-to-pn-amr", files=files)
    amr = response.json()
    """
    # NOTE: Opening the zip file mutates the object and prevents it from being reopened.
    # Since llm_proxy also needs to open the zip file, we should send a copy instead.
    linespans = await llm_proxy.get_lines_of_model(copy.deepcopy(zip_file))
    
    line_begin=[]
    line_end=[]
    files=[]
    blobs=[]
    amrs=[]
    for linespan in linespans:
        lines = linespan.block[0].split("-")
        line_begin.append(max(
            int(lines[0][1:]) - 1, 0
        ))  # Normalizing the 1-index response from llm_proxy
        line_end.append(int(lines[1][1:]))

        # Currently the llm_proxy only works on the first file in a zip_archive.
        # So we are required to do the same when slicing the source code using its output.
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in [".py"]:
                files.append(file)
                blobs.append(zip.open(file).read().decode("utf-8"))

    # The source code is a string, so to slice using the line spans, we must first convert it to a list.
    # Then we can convert it back to a string using .join
    for i in len(blobs):
        if line_begin[i] == line_end[i]:
            print("failed linespan")
        else: 
            blobs[i] = "".join(blobs[i].splitlines(keepends=True)[line_begin:line_end])
            amrs.append(await code_snippets_to_pn_amr(
                code2fn.System(
                    files=files,
                    blobs=blobs,
                    root_name=Path(zip_file.files[i]).stem,
                    system_name=Path(zip_file.files[i]).stem,
                )
            ))
    # we will return the amr with most states, in assumption it is the most "correct"
    # by default it returns the first entry
    amr = amrs[0]
    for temp_amr in amrs:
        try:
            temp_len = len(temp_amr['model']['states'])
            amr_len = len(amr['model']['states'])
            if temp_len > amr_len:
                amr = temp_amr
        except:
            continue

    return amr


""" TODO: The regnet endpoints are currently outdated
# zip archive -> fn -> regnet amr
@router.post("/code/codebase-to-rn-amr", summary="Code repo (zip archive) → RegNet AMR")
async def repo_to_rn_amr(zip_file: UploadFile = File()):
    gromet = await code2fn.fn_given_filepaths_zip(zip_file)
    res = requests.put(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE POST /models/RN failed to process payload",
                "payload": gromet,
            },
        )
    return res.json()
"""