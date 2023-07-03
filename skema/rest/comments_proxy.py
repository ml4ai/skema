# -*- coding: utf-8 -*-
"""
Proxies requests to skema_rs API to provide a unified API.
"""

from enum import Enum

from typing import Text, Union
from typing_extensions import Annotated
from skema.program_analysis.comments import CodeComments
from skema.rest.proxies import SKEMA_RS_ADDESS
from fastapi import APIRouter, FastAPI, Body, File, Response, Request, Query
from pydantic import BaseModel, Field
import os
import requests

class SupportedLanguageForCommentExtraction(str, Enum):
    python = "Python"

router = APIRouter()

class CodeSnippet(BaseModel):
    code: str = Field(
        title="code",
        description="snippet of code in referenced language",
        example={"# this is a comment\ngreet = lambda: print('howdy!')"},
    )    
    language: SupportedLanguageForCommentExtraction

@router.post("/extract-comments", response_model=CodeComments)
async def proxy_extract_comments(code_snippet: CodeSnippet) -> CodeComments:
    """
    Endpoint for extracting CodeComments JSON from a code snippet.
    Snippet must be in a supported language.
    """
    return requests.post(f"{SKEMA_RS_ADDESS}/extract-comments", json=code_snippet).json()

@router.post("/extract-comments-from-zipfile", summary=(
        "Send a zip file containing a code system comprised of supported languages,"
        " get comments for each file."
    ), response_model=CodeComments)
async def proxy_extract_comments_from_zip(request: Request) -> CodeComments:
    """
    Endpoint for generating CodeComments JSON from a zip archive of arbitrary depth and structure.
    All source files with a supported file extension (TBD) will be processed.
    """
    zip_data = await request.body()
    r = requests.post(f"{SKEMA_RS_ADDESS}/extract-comments-from-zipfile", data=zip_data)
    return r.json()