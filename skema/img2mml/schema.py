# -*- coding: utf-8 -*-
"""
Data models used by REST endpoints for eqn2mml, img2mml, and latex2mml
"""

from typing_extensions import Annotated
from pydantic import BaseModel, Field
from fastapi import File


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
    tex_src: str = Field(title="LaTeX equation", description="The LaTeX equation to process")
    class Config:
        schema_extra = {
            "example": {
                "tex_src": "\\frac{\\partial x}{\\partial t} = {\\alpha x} - {\\beta x y}",
            },
            "examples": [
                {
                  "tex_src": "\\frac{\\partial x}{\\partial t} = {\\alpha x} - {\\beta x y}"
                },
                {
                  "tex_src": "\\frac{\\partial y}{\\partial t} = {\\alpha x y} - {\\gamma y}"
                },
                {
                  "tex_src": "E = mc^{2}",
                },
                {
                  "tex_src": "\\frac{P(\\textrm{a } | \\textrm{ b}) \\times P(\\textrm{b})}{P(\\textrm{a})}",
                }
            ]
        }