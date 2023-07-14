# -*- coding: utf-8 -*-
"""
Convert the LaTeX equation to the corresponding presentation MathML using the MathJAX service.
Please run the following command to initialize the MathJAX service:
node data_generation/mathjax_server.js
"""

from typing import Text
from typing_extensions import Annotated
from fastapi import APIRouter, FastAPI, Response, Request, Query, UploadFile
from skema.rest.proxies import SKEMA_MATHJAX_ADDRESS
from skema.img2mml.api import (
    get_mathml_from_bytes,
    get_mathml_from_latex,
)
from skema.img2mml import schema
import base64
import requests
from skema.img2mml.api import Image2MathML
from pathlib import Path

cwd = Path(__file__).parents[0]
config_path = cwd / "configs" / "xfmer_mml_config.json"
vocab_path = cwd / "trained_models" / "arxiv_im2mml_with_fonts_with_boldface_vocab.txt"
model_path = (
    cwd / "trained_models" / "cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt"
)

image2mathml_db = Image2MathML(
    config_path=config_path, vocab_path=vocab_path, model_path=model_path
)

router = APIRouter()


def b64_image_to_mml(img_b64: str) -> str:
    """Helper method to convert image (encoded as base64) to MML"""
    img_bytes = base64.b64decode(img_b64)
    # convert bytes of png image to tensor:q
    return get_mathml_from_bytes(img_bytes, image2mathml_db)


EquationQueryParameter = Annotated[
    Text,
    Query(
        examples={
            "lotka eq1": {
                "summary": "Lotka-Volterra (eq1)",
                "description": "Lotka-Volterra (eq1)",
                "value": "\\frac{\\delta x}{\\delta t} = {\\alpha x} - {\\beta x y}",
            },
            "lotka eq2": {
                "summary": "Lotka-Volterra (eq2)",
                "description": "Lotka-Volterra (eq2)",
                "value": "\\frac{\\delta y}{\\delta t} = {\\alpha x y} - {\\gamma y}",
            },
            "simple": {
                "summary": "A familiar equation",
                "description": "A simple equation (mass-energy equivalence)",
                "value": "E = mc^{2}",
            },
            "complex": {
                "summary": "A more feature-rich equation (Bayes' rule)",
                "description": "A equation drawing on latex features",
                "value": "\\frac{P(\\textrm{a } | \\textrm{ b}) \\times P(\\textrm{b})}{P(\\textrm{a})}",
            },
        },
    ),
]


def process_latex_equation(eqn: Text) -> Response:
    """Helper function used by both GET and POST LaTeX equation processing endpoints"""
    res = get_mathml_from_latex(eqn)
    return Response(content=res, media_type="application/xml")


@router.get(
    "/img2mml/healthcheck",
    summary="Check health of eqn2mml service",
    response_model=int,
    status_code=200,
)
def img2mml_healthcheck() -> int:
    return 200


@router.get(
    "/latex2mml/healthcheck",
    summary="Check health of mathjax service",
    response_model=int,
    status_code=200,
)
def latex2mml_healthcheck() -> int:
    try:
        return int(requests.get(f"{SKEMA_MATHJAX_ADDRESS}/healthcheck").status_code)
    except:
        return 500


@router.post("/image/mml", summary="Get MathML representation of an equation image")
async def post_image_to_mathml(data: UploadFile) -> Response:
    """
    Endpoint for generating MathML from an input image.

    ### Python example
    ```
    import requests

    files = {
      "data": open("bayes-rule-white-bg.png", "rb"),
    }
    r = requests.post("http://0.0.0.0:8000/image/mml", files=files)
    print(r.text)
    """
    # Read image data
    image_bytes = await data.read()

    # pass image bytes to get_mathml_from_bytes function
    res = get_mathml_from_bytes(image_bytes, image2mathml_db)

    return Response(content=res, media_type="application/xml")


@router.post(
    "/image/base64/mml", summary="Get MathML representation of an equation image"
)
async def post_b64image_to_mathml(request: Request) -> Response:
    """
    Endpoint for generating MathML from an input image.

    ### Python example
    ```
    from pathlib import Path
    import base64
    import requests

    url = "http://0.0.0.0:8000/image/base64/mml"
    with Path("bayes-rule-white-bg.png").open("rb") as infile:
      img_bytes = infile.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    r = requests.post(url, data=img_b64)
    print(r.text)
    """
    img_b64 = await request.body()
    res = b64_image_to_mml(img_b64)
    return Response(content=res, media_type="application/xml")


@router.get("/latex/mml", summary="Get MathML representation of a LaTeX equation")
async def get_tex_to_mathml(tex_src: EquationQueryParameter) -> Response:
    """
    GET endpoint for generating MathML from an input LaTeX equation.

    ### Python example
    ```
    import requests

    r = requests.get("http://0.0.0.0:8000/latex/mml", params={"tex_src":"E = mc^{c}"})
    print(r.text)
    """
    return process_latex_equation(tex_src)


@router.post("/latex/mml", summary="Get MathML representation of a LaTeX equation")
async def post_tex_to_mathml(eqn: schema.LatexEquation) -> Response:
    """
    Endpoint for generating MathML from an input LaTeX equation.

    ### Python example
    ```
    import requests

    r = requests.post("http://0.0.0.0:8000/latex/mml", json={"tex_src":"E = mc^{2}"})
    print(r.text)
    """
    # convert latex string to presentation mathml
    return process_latex_equation(eqn.tex_src)


app = FastAPI()
app.include_router(router)
