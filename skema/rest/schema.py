# -*- coding: utf-8 -*-
"""
Response models for API
"""

from pydantic import BaseModel, Field
from typing import List
# see https://github.com/pydantic/pydantic/issues/5821#issuecomment-1559196859
from typing_extensions import Literal
from skema.img2mml import eqn2mml


class HealthStatus(BaseModel):
    morae: int = Field(description="HTTP status code for MORAE service", ge=100, le=599)
    mathjax: int = Field(
        description="HTTP status code for mathjax service (used by eqn2mml latex2mml endpoints)",
        ge=100,
        le=599,
    )
    eqn2mml: int = Field(
        description="HTTP status code for eqn2mml service (img2mml endpoints)",
        ge=100,
        le=599,
    )
    code2fn: int = Field(
        description="HTTP status code for code2fn service", ge=100, le=599
    )


class EquationImagesToAMR(BaseModel):
    # FIXME: will this work or do we need base64?
    images: List[eqn2mml.ImageBytes]
    model: Literal["regnet", "petrinet"] = Field(description="The model type")


class EquationLatexToAMR(BaseModel):
    equations: List[str] = Field(description="Equations in LaTeX")
    model: Literal["regnet", "petrinet"] = Field(description="The model type")

class CodeSnippet(BaseModel):
    code: str = Field(
        title="code",
        description="snippet of code in referenced language",
        example="# this is a comment\ngreet = lambda: print('howdy!')",
    )
    language: Literal["Python", "Fortran", "CppOrC"] = Field(
        title="language", 
        description="Programming language corresponding to `code`"
    )

class TextReadingInputDocuments(BaseModel):
    """ Model of text reading request body """
    texts: List[str]