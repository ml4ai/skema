# -*- coding: utf-8 -*-
"""
End-to-end skema workflows
"""


from typing import List
from typing_extensions import Annotated
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.rest import schema
from skema.img2mml import eqn2mml
from skema.skema_py import server as code2fn
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field
import requests


router = APIRouter()


# equation images -> mml -> amr
@router.post("/images/equations-to-amr", summary="Equations (images) → MML → AMR")
async def equations_to_amr(data: schema.EquationImagesToAMR):
    """
    Converts images of equations to AMR.

    ### Python example
    ```
    import requests

    images = [open("eq1.png", "rb").read(), open("eq2.png", "rb").read()]
    }
    url = "0.0.0.0"
    r = requests.post(f"{url}/workflows/image/eqn-to-amr", json={"images": images, "model": "regnet"})
    r.json()
    """
    mml: List[str] = [eqn2mml.post_image_to_mathml(img).content for img in data.images]
    return requests.post(
        f"{SKEMA_RS_ADDESS}/mathml/amr", json={"mathml": mml, "model": data.model}
    ).json()


# tex equations -> pmml -> amr
@router.post("/latex/equations-to-amr", summary="Equations (LaTeX) → pMML → AMR")
async def equations_to_amr(data: schema.EquationLatexToAMR):
    """
    Converts equations (in LaTeX) to AMR.

    ### Python example
    ```
    import requests

    equations = ["x = 2", "y = 3"]
    }
    url = "0.0.0.0"
    r = requests.post(f"{url}/workflows/latex/equations-to-amr", json={"equations": equations, "model": "regnet"})
    r.json()
    """
    mml: List[str] = [
        eqn2mml.post_tex_to_mathml(eqn2mml.LatexEquation(tex_src=tex)).content
        for tex in data.equations
    ]
    return requests.post(
        f"{SKEMA_RS_ADDESS}/mathml/amr", json={"mathml": mml, "model": data.model}
    ).json()


# code snippets -> fn -> petrinet amr
@router.post("/code/snippets-to-pn-amr", summary="Code snippets → PetriNet AMR")
async def snippets_to_amr(system: code2fn.System):
    if system.comments == None:
        # FIXME: get comments
        pass
    gromet = await code2fn.fn_given_filepaths(system)
    print(f"gromet:\t{gromet}")
    return requests.post(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet).json()


# code snippets -> fn -> regnet amr
@router.post("/code/snippets-to-rn-amr", summary="Code snippets → RegNet AMR")
async def snippets_to_amr(system: code2fn.System):
    if system.comments == None:
        # FIXME: get comments and produce another system
        pass
    gromet = await code2fn.fn_given_filepaths(system)
    return requests.post(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet).json()


# zip archive -> fn -> petrinet amr
@router.post(
    "/code/codebase-to-pn-amr", summary="Code repo (zip archive) → PetriNet AMR"
)
async def repo_to_amr(zip_file: UploadFile = File()):
    # FIXME: get comments
    gromet = code2fn.fn_given_filepaths_zip(zip_file)
    return requests.post(f"{SKEMA_RS_ADDESS}/models/PN", json=gromet).json()


# zip archive -> fn -> petrinet amr
@router.post("/code/codebase-to-pn-amr", summary="Code repo (zip archive) → RegNet AMR")
async def repo_to_amr(zip_file: UploadFile = File()):
    # FIXME: get comments
    gromet = code2fn.fn_given_filepaths_zip(zip_file)
    return requests.post(f"{SKEMA_RS_ADDESS}/models/RN", json=gromet).json()
