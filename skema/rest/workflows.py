# -*- coding: utf-8 -*-
"""
End-to-end skema workflows
"""
import copy
import time
from zipfile import ZipFile
from io import BytesIO
from typing import List
from pathlib import Path
import httpx
import requests

from fastapi import APIRouter, Depends, File, UploadFile, FastAPI, Request
from starlette.responses import JSONResponse

from skema.img2mml import eqn2mml
from skema.img2mml.eqn2mml import image2mathml_db, b64_image_to_mml
from skema.img2mml.api import get_mathml_from_bytes
from skema.isa.lib import generate_code_graphs, align_eqn_code, convert_to_dict
from skema.rest import config, schema, utils, llm_proxy
from skema.rest.equation_extraction import process_pdf_and_images
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.skema_py import server as code2fn


router = APIRouter()


# equations [mathml, latex] -> amrs [Petrinet, Regnet, GAMR, MET, Decapode]
@router.post(
    "/consolidated/equations-to-amr", summary="equations [mathml, latex] → AMRs [Petrinet, Regnet, GAMR, MET, Decapode]"
)
async def equation_to_amrs(data: schema.EquationsToAMRs, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts equations (in LaTeX or MathML) to an AMR (Petrinet, Regnet, GAMR, MET, or Decapode). 
    
    ## If Petrinet or Regnet is selected and the conversion fails, we fall back to converting to a Generalized AMR. 
    ---
    ### Python example
    ```
    import requests

    equations = [
      "\\frac{\\delta x}{\\delta t} = {\\alpha x} - {\\beta x y}",
      "\\frac{\\delta y}{\\delta t} = {\\alpha x y} - {\\gamma y}"
    ]
    url = "0.0.0.0"
    r = requests.post(f"{url}/consolidated/equations-to-amr", json={"equations": equations, "model": "regnet"})
    r.json()
    ```
    ---
    parameters:
      - name: equations
        - schema:
          - type: array
          - items:
            - type: string
        - required: true
        - description: This is a list of equations, in either pMathML or LaTeX
      - name: model
        - type: string
        - required: true
        - description: This specifies the type of model that the output AMR will be in
        - examples:
          - Petrinet:
            - summary: For making a petrinet
            - value: "petrinet"
          - Regnet:
            - summary: For making a regnet
            - value: "regnet"
          - Decapode:
            - summary: For making a decapode
            - value: "decapode"
          - Generalized AMR:
            - summary: For making a generalized AMR
            - value: "gamr"
          - Math Expression Tree:
            - summary: For making a Math Expression Tree
            - value: "met"
    """
    eqns = utils.parse_equations(data.equations)
    if data.model == "petrinet" or data.model == "regnet":
        payload = {"mathml": eqns, "model": data.model}
        res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
        if res.status_code != 200:
            res_new = await client.put(f"{SKEMA_RS_ADDESS}/mathml/g-amr", json=eqns)
            if res_new.status_code != 200:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": f"Attempted creation of {data.model} AMR, which failed. Then tried creation of Generalized AMR, which also failed with the following error {res_new.text}. Please check equations, seen as pMathML below.",
                        "payload": eqns,
                    },
                )
            res = res_new
    elif data.model == "met":
        res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/met", json=eqns)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"PUT /mathml/met failed to process payload with error {res.text}",
                    "payload": eqns,
                },
            )
    elif data.model == "gamr":
        res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/g-amr", json=eqns)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"PUT /mathml/met failed to process payload with error {res.text}",
                    "payload": eqns,
                },
            )
    elif data.model == "decapode":
        res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/decapodes", json=eqns)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"PUT /mathml/met failed to process payload with error {res.text}",
                    "payload": eqns,
                },
            )
    else:
        return JSONResponse(
            status_code=401,
            content={
                "error": f"{data.model} is not a supported model type",
                "payload": eqns,
            },
        )


    return res.json()
    
# Code Snippets -> amrs [Petrinet, Regnet, GAMR, MET]
@router.post(
    "/consolidated/code-snippets-to-amrs", summary="code snippets → AMRs [Petrinet, Regnet, GAMR, MET]"
)
async def code_snippets_to_amrs(system: code2fn.System, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Code snippets to AMR workflow. This endpoint takes a code snippet, assumed to contain dynamics, and extracts the 
    Math Expression Tree of the dynamics, which is then converted into an AMR of the specified type.

    ### Python example
    ```
    import requests

    # Single file
    single_snippet_payload = {"files": ["code.py"], "blobs": ["def sir(s: float, i: float, r: float, beta: float, gamma: float, n: float) -> Tuple[float, float, float]:\n    \"\"\"The SIR model, one time step.\"\"\"\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale"], "model": "petrinet"}

    response = requests.post("http://0.0.0.0:8000/workflows/consolidated/code-snippets-to-amrs", json=single_snippet_payload)
    gromet_json = response.json()
    ```
    """
    gromet = await code2fn.fn_given_filepaths(system)
    gromet, _ = utils.fn_preprocessor(gromet)
    if system.model == "petrinet":
        res = await client.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"MORAE PUT /models/PN failed to process payload ({res.text})",
                    "payload": gromet,
                },
            )
    elif system.model == "regnet":
        res = await client.put(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"MORAE PUT /models/RN failed to process payload ({res.text})",
                    "payload": gromet,
                },
            )
    elif system.model == "met":
        res = await client.put(f"{SKEMA_RS_ADDESS}/models/MET", json=gromet)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"MORAE PUT /models/MET failed to process payload ({res.text})",
                    "payload": gromet,
                },
            )
    elif system.model == "gamr":
        res = await client.put(f"{SKEMA_RS_ADDESS}/models/G-AMR", json=gromet)
        if res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"MORAE PUT /models/G-AMR failed to process payload ({res.text})",
                    "payload": gromet,
                },
            )
    else:
        return JSONResponse(
            status_code=401,
            content={
                "error": f"{system.model} is not a supported model type",
                "payload": gromet,
            },
        )
    return res.json()


# equation images -> mml -> amr
@router.post(
    "/images/base64/equations-to-amr", summary="Equations (base64 images) → MML → AMR"
)
async def equations_img_to_amr(data: schema.EquationImagesToAMR, client: httpx.AsyncClient = Depends(utils.get_client)):
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
    ```
    """
    mml: List[str] = [
        utils.clean_mml(eqn2mml.b64_image_to_mml(img)) for img in data.images
    ]
    payload = {"mathml": mml, "model": data.model}
    # FIXME: why is this a PUT?
    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
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
async def equations_img_to_latex(data: UploadFile, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts images of equations to LaTeX.

    ### Python example

    Endpoint for generating LaTeX from an input image.

    ```
    import requests
    import json

    files = {
      "data": open("bayes-rule-white-bg.png", "rb"),
    }
    r = requests.post("http://0.0.0.0:8000/workflows/images/equations-to-latex", files=files)
    print(json.loads(r.text))
    ```
    """
    # Read image data
    image_bytes = await data.read()

    # pass image bytes to get_mathml_from_bytes function
    mml_res = get_mathml_from_bytes(image_bytes, image2mathml_db)
    proxy_url = f"{SKEMA_RS_ADDESS}/mathml/latex"
    print(f"MMML:\t{mml_res}")
    print(f"Proxying request to {proxy_url}")
    response = await client.post(proxy_url, data=mml_res)
    # Check the response
    if response.status_code == 200:
        # The request was successful
        return response.text
    else:
        # The request failed
        print(f"Error: {response.status_code}")
        print(response.text)
        return f"Error: {response.status_code} {response.text}"


# equation images -> base64 -> mml -> latex
@router.post("/images/base64/equations-to-latex", summary="Equations (images) → MML → LaTeX")
async def equations_to_latex(request: Request, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts images of equations to LaTeX.

    ### Python example

    Endpoint for generating LaTeX from an input image.

    ```
    from pathlib import Path
    import base64
    import requests

    url = "http://127.0.0.1:8000/workflows/images/base64/equations-to-latex"
    with Path("test.png").open("rb") as infile:
      img_bytes = infile.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    r = requests.post(url, data=img_b64)
    print(r.text)
    ```
    """
    # Read image data
    img_b64 = await request.body()
    mml_res = b64_image_to_mml(img_b64)

    # pass image bytes to get_mathml_from_bytes function
    proxy_url = f"{SKEMA_RS_ADDESS}/mathml/latex"
    print(f"MML:\t{mml_res}")
    print(f"Proxying request to {proxy_url}")
    response = await client.post(proxy_url, data=mml_res)
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
async def lx_equations_to_amr(data: schema.EquationLatexToAMR, client: httpx.AsyncClient = Depends(utils.get_client)):
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
    ```
    """
    mml: List[str] = [
        utils.clean_mml(eqn2mml.get_mathml_from_latex(tex)) for tex in data.equations
    ]
    payload = {"mathml": mml, "model": data.model}
    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
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
async def equations_to_amr(data: schema.MmlToAMR, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts equations (in LaTeX or pMathML) to MathExpressionTree (JSON).

    ### Python example
    ```
    import requests

    payload = {
        "equations": 
        [
            "<math><mfrac><mrow><mi>d</mi><mi>E</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>&#x03B2;</mi><mi>I</mi><mi>S</mi><mo>&#x2212;</mo><mi>&#x03B4;</mi><mi>E</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>R</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>(1&#x2212;&#x03B1;)</mi><mi>&#x03B3;</mi><mi>I</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>I</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>&#x03B4;</mi><mi>E</mi><mo>&#x2212;</mo><mi>(1&#x2212;&#x03B1;)</mi><mi>&#x03B3;</mi><mi>I</mi><mo>&#x2212;</mo><mi>&#x03B1;</mi><mi>&#x03C1;</mi><mi>I</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>D</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mi>&#x03B1;</mi><mi>&#x03C1;</mi><mi>I</mi></math>",
            "<math><mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mo>&#x2212;</mo><mi>&#x03B2;</mi><mi>I</mi><mi>S</mi></math>"
        ],
        "model": "petrinet"
    }

    url = "http://127.0.0.1:8000"

    r = requests.post(f"{url}/workflows/pmml/equations-to-amr",  json=payload)
    print(r.json())
    ```
    """
    payload = {"mathml": data.equations, "model": data.model}
    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/amr", json=payload)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /mathml/amr failed to process payload with error {res.text}",
                "payload": payload,
            },
        )
    return res.json()


# equations(pmml or latex) -> MathExpressionTree
@router.post("/equations-to-met", summary="Equations (LaTeX/pMML) → MathExpressionTree")
async def equations_to_met(data: schema.EquationToMET, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts equations (in LaTeX or pMathML) to MathExpressionTree (JSON).

    ### Python example
    ```
    import requests

    equations = [
        "E=mc^2",
        "c=\\frac{a}{b}"
    ]

    url = "http://127.0.0.1:8000"

    r = requests.post(f"{url}/workflows/equations-to-met",  json={"equations": equations})
    print(r.json())
    ```
    """
    eqns = utils.parse_equations(data.equations)

    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/met", json=eqns)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"PUT /mathml/met failed to process payload with error {res.text}",
                "payload": eqns,
            },
        )
    return res.json()


# equations(pmml or latex) -> Generalized AMR
@router.post("/equations-to-gamr", summary="Equations (LaTeX/pMML) → Generalized AMR")
async def equations_to_gamr(data: schema.EquationToMET, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Converts equations (in LaTeX or pMathML) to Generalized AMR (JSON).

    ### Python example
    ```
    import requests

    equations = [
        "E=mc^2",
        "c=\\frac{a}{b}"
    ]

    url = "http://127.0.0.1:8000"

    r = requests.post(f"{url}/workflows/equations-to-gamr",  json={"equations": equations})
    print(r.json())
    ```
    """
    eqns = utils.parse_equations(data.equations)

    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/g-amr", json=eqns)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"PUT /mathml/met failed to process payload with error {res.text}",
                "payload": eqns,
            },
        ) 
    return res.json()


# code snippets -> fn -> petrinet amr
@router.post("/code/snippets-to-pn-amr", summary="Code snippets → PetriNet AMR")
async def code_snippets_to_pn_amr(system: code2fn.System, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Code snippets to Petrinet AMR workflow. This endpoint takes a code snippet, assumed to contain dynamics, and extracts the 
    Math Expression Tree of the dynamics, which is then converted into a Petrinet AMR.

    ### Python example
    ```
    import requests

    # Single file
    single_snippet_payload = {"files": ["code.py"], "blobs": ["def sir(s: float, i: float, r: float, beta: float, gamma: float, n: float) -> Tuple[float, float, float]:\n    \"\"\"The SIR model, one time step.\"\"\"\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale"],}

    response = requests.post("http://0.0.0.0:8000/workflows/code/snippets-to-met", json=single_snippet_payload)
    gromet_json = response.json()
    ```
    """
    gromet = await code2fn.fn_given_filepaths(system)
    gromet, _ = utils.fn_preprocessor(gromet)
    # print(f"gromet:{gromet}")
    # print(f"client.follow_redirects:\t{client.follow_redirects}")
    # print(f"client.timeout:\t{client.timeout}")
    res = await client.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload ({res.text})",
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
async def repo_to_pn_amr(zip_file: UploadFile = File(), client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Codebase to AMR workflow. This endpoint uses an a simple algorithm to identify the dynamics and then we slice 
    that portion of the code to extract dynamics from it.

    ### Python example
    ```
    import requests

    files = {
        'zip_archive': open('model_source.zip')
    }
    response = requests.post("localhost:8000/workflows/code/codebase-to-pn-amr", files=files)
    amr = response.json()
    ```
    """
    gromet = await code2fn.fn_given_filepaths_zip(zip_file)
    gromet, _ = utils.fn_preprocessor(gromet)
    res = await client.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
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
async def llm_assisted_codebase_to_pn_amr(zip_file: UploadFile = File(), client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Codebase to AMR workflow. This endpoint uses an LLM to identify the dynamics and then we slice 
    that portion of the code to extract dynamics from it.

    ### Python example
    ```
    import requests

    files = {
        'zip_archive': open('model_source.zip')
    }
    response = requests.post("localhost:8000/workflows/code/llm-assisted-codebase-to-pn-amr", files=files)
    amr = response.json()
    ```
    """
    # NOTE: Opening the zip file mutates the object and prevents it from being reopened.
    # Since llm_proxy also needs to open the zip file, we should send a copy instead.
    print(f"Time call linespan: {time.time()}")
    linespans = await llm_proxy.get_lines_of_model(copy.deepcopy(zip_file))
    print(f"Time response linespan: {time.time()}")

    line_begin = []
    import_begin = []
    line_end = []
    import_end = []
    files = []
    blobs = []
    amrs = []

    # There could now be multiple blocks that we need to handle and adjoin together
    for linespan in linespans:
        blocks = len(linespan.block)
        lines = linespan.block[blocks-1].split("-")
        line_begin.append(
            max(int(lines[0][1:]) - 1, 0)
        )  # Normalizing the 1-index response from llm_proxy
        line_end.append(int(lines[1][1:]))
        if blocks == 2:
            lines = linespan.block[0].split("-")
            import_begin.append(
            max(int(lines[0][1:]) - 1, 0)
            )  # Normalizing the 1-index response from llm_proxy
            import_end.append(int(lines[1][1:]))

        # So we are required to do the same when slicing the source code using its output.
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in [".py"]:
                # Skip file if located in a hidden directory or MACOSX artifact
                valid = True
                for parent in file_obj.parents:
                    if parent.name == "_MACOSX":
                        valid = False
                        break
                    elif parent.name.startswith("."):
                        valid = False
                        break 
                if valid:
                    files.append(file)
                    blobs.append(zip.open(file).read().decode("utf-8"))

    # The source code is a string, so to slice using the line spans, we must first convert it to a list.
    # Then we can convert it back to a string using .join
    logging = []
    import_counter = 0
    for i in range(len(blobs)):
        if line_begin[i] == line_end[i]:
            print("failed linespan")
        else:
            if len(linespans[i].block) == 2:
                temp = "".join(blobs[i].splitlines(keepends=True)[import_begin[import_counter]:import_end[import_counter]])
                blobs[i] = temp + "\n" + "".join(blobs[i].splitlines(keepends=True)[line_begin[i]:line_end[i]])
                import_counter += 1
            else:
                blobs[i] = "".join(blobs[i].splitlines(keepends=True)[line_begin[i]:line_end[i]])
            try:
                print(f"Time call code-snippets: {time.time()}")
                gromet = await code2fn.fn_given_filepaths(code2fn.System(
                             files=[files[i]],
                             blobs=[blobs[i]],
                         ))
                gromet, _ = utils.fn_preprocessor(gromet)
                code_snippet_response = await client.put(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet)
                code_snippet_response = code_snippet_response.json()
                print(f"Time response code-snippets: {time.time()}")
                if "model" in code_snippet_response:
                    code_snippet_response["header"]["name"] = "LLM-assisted code to amr model"
                    code_snippet_response["header"]["description"] = f"This model came from code file: {files[i]}"
                    code_snippet_response["header"]["linespan"] = f"{linespans[i]}"
                    amrs.append(code_snippet_response)
                else:
                    print("snippets failure")
                    logging.append(f"{files[i]} failed to parse an AMR from the dynamics")
            except Exception as e:
                print("Hit except to snippets failure")
                print(f"Exception:\t{e}")
                logging.append(f"{files[i]} failed to parse an AMR from the dynamics")
    # we will return the amr with most states, in assumption it is the most "correct"
    # by default it returns the first entry
    print(f"{amrs}")
    try:
        amr = amrs[0]
        for temp_amr in amrs:
            try:
                temp_len = len(temp_amr["model"]["states"])
                amr_len = len(amr["model"]["states"])
                if temp_len > amr_len:
                    amr = temp_amr
            except:
                continue
    except:
        amr = logging

    return amr

# code snippets -> fn -> MET
@router.post("/code/snippets-to-met", summary="Code snippets → MET")
async def code_snippets_to_MET(system: code2fn.System, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Code snippets to Math Expression Tree workflow. This endpoint takes a code snippet, assumed to contain dynamics, and extracts the 
    Math Expression Tree of the dynamics. 

    ### Python example
    ```
    import requests

    # Single file
    single_snippet_payload = {"files": ["code.py"], "blobs": ["def sir(s: float, i: float, r: float, beta: float, gamma: float, n: float) -> Tuple[float, float, float]:\n    \"\"\"The SIR model, one time step.\"\"\"\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale"],}

    response = requests.post("http://0.0.0.0:8000/workflows/code/snippets-to-met", json=single_snippet_payload)
    gromet_json = response.json()
    ```
    """
    gromet = await code2fn.fn_given_filepaths(system)
    gromet, _ = utils.fn_preprocessor(gromet)
    # print(f"gromet:{gromet}")
    # print(f"client.follow_redirects:\t{client.follow_redirects}")
    # print(f"client.timeout:\t{client.timeout}")
    res = await client.put(f"{SKEMA_RS_ADDESS}/models/MET", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload ({res.text})",
                "payload": gromet,
            },
        )
    return res.json()

# code snippets -> fn -> generalized amr
@router.post("/code/snippets-to-gamr", summary="Code snippets → Generalized-AMR")
async def code_snippets_to_G_AMR(system: code2fn.System, client: httpx.AsyncClient = Depends(utils.get_client)):
    """
    Code snippets to Generalized AMR workflow. This endpoint takes a code snippet, assumed to contain dynamics, and extracts the 
    Math Expression Tree of the dynamics and then converts that to our Generalized AMR represenation. 

    ### Python example
    ```
    import requests

    # Single file
    single_snippet_payload = {"files": ["code.py"], "blobs": ["def sir(s: float, i: float, r: float, beta: float, gamma: float, n: float) -> Tuple[float, float, float]:\n    \"\"\"The SIR model, one time step.\"\"\"\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale"],}
    
    response = requests.post("http://0.0.0.0:8000/workflows/code/snippets-to-gamr", json=single_snippet_payload)
    gromet_json = response.json()
    ```
    """
    gromet = await code2fn.fn_given_filepaths(system)
    gromet, _ = utils.fn_preprocessor(gromet)
    # print(f"gromet:{gromet}")
    # print(f"client.follow_redirects:\t{client.follow_redirects}")
    # print(f"client.timeout:\t{client.timeout}")
    res = await client.put(f"{SKEMA_RS_ADDESS}/models/G-AMR", json=gromet)
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload ({res.text})",
                "payload": gromet,
            },
        )
    return res.json()


# code snippets -> fn -> Vec<MET> -> alignment result
#              mathml ->    MET   ->
@router.post("/isa/code-eqn-align", summary="ISA aided inference")
async def code_snippets_to_isa_align(
    mml_system: code2fn.MML_System,
    client: httpx.AsyncClient = Depends(utils.get_client)
):
    """
    Endpoint for ISA aided inference.

    Args:
        mml_system (code2fn.MML_System): Input data containing MML and system details.
        client (httpx.AsyncClient): An asynchronous HTTP client dependency.

    Returns:
        JSONResponse: Response containing aligned equation and code information.
        # The dictionary of the following data structure
        # matching_ratio: the matching ratio between the equations 1 and the equation 2
        # num_diff_edges: the number of different edges between the equations 1 and the equation 2
        # node_labels1: the name list of the variables and terms in the equation 1
        # node_labels2: the name list of the variables and terms in the equation 2
        # aligned_indices1: the aligned indices in the name list of the equation 1 (-1 means missing)
        # aligned_indices2: the aligned indices in the name list of the equation 2 (-1 means missing)
        # union_graph: the visualization of the alignment result
        # perfectly_matched_indices1: strictly matched node indices in Graph 1

    Raises:
        HTTPException: If there are errors in processing the payload or communication with external services.

    Note:
        This endpoint takes MML information and system details, processes the data, and communicates with external services
        to perform ISA aided inference.

    """
    # Extracting system details using code2fn module
    gromet = await code2fn.fn_given_filepaths(mml_system.system)
    gromet, _ = utils.fn_preprocessor(gromet)

    # Sending processed data to an external service
    res = await client.put(f"{SKEMA_RS_ADDESS}/models/MET", json=gromet)

    # Checking the response status and handling errors if any
    if res.status_code != 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"MORAE PUT /models/PN failed to process payload ({res.text})",
                "payload": gromet,
            },
        )
    else:
        # Further processing and communication with the code-exp-graphs service
        code_graph_res = await client.put(
            f"{SKEMA_RS_ADDESS}/mathml/code-exp-graphs", json=res.json()
        )

        # Checking the response status and handling errors if any
        if code_graph_res.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"code-exp-graphs PUT mathml/code-exp-graphs failed to process payload ({res.json()})",
                    "payload": res.json(),
                },
            )

        # Aligning equation and code
        alignment_res = align_eqn_code(utils.clean_mml(mml_system.mml), code_graph_res.text)

        # Converting numpy arrays to dictionaries for deserialization
        converted_alignment_res = convert_to_dict(alignment_res)

        # Returning the final aligned result
        return JSONResponse(
            status_code=200,
            content=converted_alignment_res,
        )


# PDF -> [COSMOS] -> Equation images -> [MIT Service] -> Equation JSON
@router.post(
    "/equations_extraction",
    summary="PDF -> [COSMOS] -> Equation images -> [MIT Service] -> Equation JSON",
)
async def equations_extraction(
    equation_extraction_system: code2fn.Equation_Extraction_System,
):
    """
    Extracts images of equations from PDF and Generates the JSON info.

    ### Python example

    Endpoint for extracting images of equations from PDF.

    ```
    import requests

    # Define the URL for the API endpoint
    url: str = "http://127.0.0.1:8000/workflows/equations_extraction"

    # Specify the local path to the PDF file
    pdf_local_path: str = "your PDF path"

    # Specify the folder path where images will be saved
    save_folder: str = "your save folder path"

    # Specify your OpenAI API key
    gpt_key: str = "your openai key here"

    # Send a POST request to the API endpoint
    response = requests.post(url, json={"pdf_local_path": pdf_local_path, "save_folder": save_folder, "gpt_key": gpt_key})
    ```
    """
    try:
        process_pdf_and_images(
            equation_extraction_system.pdf_local_path,
            equation_extraction_system.save_folder,
            equation_extraction_system.gpt_key,
        )
        return JSONResponse(
            status_code=200,
            content={"message": "Images extracted and processed successfully."},
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )


app = FastAPI()
app.include_router(router)