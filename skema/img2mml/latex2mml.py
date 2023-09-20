# -*- coding: utf-8 -*-
"""
Convert the LaTeX equation to the corresponding presentation MathML using the MathJAX service.
Please run the following command to initialize the MathJAX service:
node data_generation/mathjax_server.js
"""

from typing import Text
from fastapi import FastAPI
from skema.img2mml.api import get_mathml_from_latex
from skema.img2mml import schema

# Create a web app using FastAPI

app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The latex2mml service is running."


@app.get("/get-mml", summary="Get MathML representation of a LaTeX equation")
async def get_mathml(tex_src: str):
    """
    GET endpoint for generating MathML from an input LaTeX equation.
    """
    # convert latex string to presentation mathml
    print(tex_src)
    return get_mathml_from_latex(tex_src)

@app.post("/latex2mml", summary="Get MathML representation of a LaTeX equation")
async def mathml(eqn: schema.LatexEquation):
    """
    Endpoint for generating MathML from an input LaTeX equation.
    """
    # convert latex string to presentation mathml
    return get_mathml_from_latex(eqn.tex_src)
