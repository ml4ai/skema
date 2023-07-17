# -*- coding: utf-8 -*-
"""
Proxies requests to skema_rs API to provide a unified API.
"""

from skema.program_analysis.comments import CodeComments
from skema.rest.proxies import SKEMA_RS_ADDESS
from skema.rest import schema
from fastapi import APIRouter, Request
import requests


router = APIRouter()


@router.post("/extract-comments", response_model=CodeComments)
async def proxy_extract_comments(code_snippet: schema.CodeSnippet) -> CodeComments:
    """
    Endpoint for extracting CodeComments JSON from a code snippet.
    Snippet must be in a supported language.
    """
    return requests.post(
        # NOTE: .dict() -> .model_dump() in Pydantic v2 (see also https://github.com/tiangolo/fastapi/issues/9710)
        f"{SKEMA_RS_ADDESS}/extract-comments",
        json=code_snippet.dict(),
    ).json()


@router.post(
    "/extract-comments-from-zipfile",
    summary=(
        "Send a zip file containing a code system comprised of supported languages,"
        " get comments for each file."
    ),
    response_model=CodeComments,
)
async def proxy_extract_comments_from_zip(request: Request) -> CodeComments:
    """
    Endpoint for generating CodeComments JSON from a zip archive of arbitrary depth and structure.
    All source files with a supported file extension (TBD) will be processed.
    """
    zip_data = await request.body()
    r = requests.post(f"{SKEMA_RS_ADDESS}/extract-comments-from-zipfile", data=zip_data)
    return r.json()