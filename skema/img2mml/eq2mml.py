# -*- coding: utf-8 -*-
"""
Convert the LaTeX equation to the corresponding presentation MathML using the MathJAX service.
Please run the following command to initialize the MathJAX service:
node data_generation/mathjax_server.js
"""

from typing import Text, Union
from typing_extensions import Annotated
from fastapi import Body, FastAPI, File, Request, Query
from skema.img2mml.api import (get_mathml_from_bytes, get_mathml_from_latex)
from pydantic import BaseModel, Field
#from skema.data.eq2mml import img_path_bayes_rule_eqn
#import base64


EquationQueryParameter = Annotated[
  Text,
  Query(
      examples={
          "simple": {
              "summary": "A familiar equation",
              "description": "A simple equation (mass-energy equivalence)",
              "value": "E = mc^{2}",
          },
          "complex": {
              "summary": "A more feature-rich equation (Bayes' rule)",
              "description": "A equation drawing on latex features",
              "value": "\\frac{P(\\textrm{a } | \\textrm{ b}) \\times P(\\textrm{b})}{P(\\textrm{a})}",
          }
      },
  ),
]

ImageBytes = Annotated[
  bytes,
  File(
      description="bytes of a PNG of an equation",
      # examples={
      #     "bayes-rule": {
      #         "summary": "PNG of Bayes' rule",
      #         "description": "PNG of Bayes' rule",
      #         "value": str(img_path_bayes_rule_eqn),
      #     }
      # },
  ),
]

class LatexEquation(BaseModel):
    tex_src: Text = Field(description="The LaTeX equation to process")
    class Config:
        schema_extra = {
            "example": {
                "tex_src": "E = mc^{c}",
            }
        }


app = FastAPI()

# FIXME: have this test the mathjax endpoint (and perhaps check the pt model loaded)
@app.get("/healthcheck", summary="Ping endpoint to test health of service", response_model=Text, status_code=200)
def ping():
    return "The eq2mml service is running."

@app.post("/image/mml", summary="Get MathML representation of an equation image", response_model=Text)
async def image_mathml(data: ImageBytes) -> Text:
    """
    Endpoint for generating MathML from an input image.
    """
    # convert bytes of png image to tensor
    print(data)
    return get_mathml_from_bytes(data)

@app.get("/latex/mml", summary="Get MathML representation of a LaTeX equation", response_model=Text)
async def tex_to_mathml(tex_src: EquationQueryParameter) -> Text:
    """
    GET endpoint for generating MathML from an input LaTeX equation.
    """
    return get_mathml_from_latex(tex_src)

@app.post("/latex/mml", summary="Get MathML representation of a LaTeX equation")
async def mathml(eqn: LatexEquation):
    """
    Endpoint for generating MathML from an input LaTeX equation.
    """
    # convert latex string to presentation mathml
    return get_mathml_from_latex(eqn.tex_src)
