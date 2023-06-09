# -*- coding: utf-8 -*-
"""
Convert the LaTeX equation to the corresponding presentation MathML using the MathJAX service.
Please run the following command to initialize the MathJAX service:
node data_generation/mathjax_server.js
"""

from fastapi import FastAPI, File
from skema.img2mml.api import get_mathml_from_latex


# Create a web app using FastAPI

app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The latex2mml service is running."


@app.put("/get-mml", summary="Get MathML representation of a LaTeX equation")
async def get_mathml(eqn: str):
    """
    Endpoint for generating MathML from an input LaTeX equation.
    """
    # convert latex string to presentation mathml
    return get_mathml_from_latex(eqn)
