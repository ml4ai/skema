# -*- coding: utf-8 -*-
"""
Proxies requests to skema_rs API to provide a unified API.
"""


from typing import Any, Dict, List, Text
from skema.rest.proxies import SKEMA_RS_ADDESS
from fastapi import APIRouter
import requests


router = APIRouter()


# FIXME: make GrometFunctionModuleCollection a pydantic model via code gen
@router.post("/model", summary="Pushes gromet (function network) to the graph database", include_in_schema=False)
async def post_model(gromet: Dict[Text, Any]):
    return requests.post(f"{SKEMA_RS_ADDESS}/models", json=gromet).json()


@router.get("/models", summary="Gets function network IDs from the graph database")
async def get_models() -> List[int]:
    request = requests.get(f"{SKEMA_RS_ADDESS}/models")
    print(f"request: {request}")
    return request.json()


@router.get("/ping", summary="Status of MORAE service")
async def healthcheck() -> str:
    return requests.get(f"{SKEMA_RS_ADDESS}/ping").text


@router.get("/version", summary="Status of MORAE service")
async def versioncheck() -> str:
    return requests.get(f"{SKEMA_RS_ADDESS}/version").text

@router.post("/mathml/decapodes", summary="Gets Decapodes from a list of MathML strings")
async def get_decapodes(mathml: List[str]) -> Dict[Text, Any]:
    return requests.put(f"{SKEMA_RS_ADDESS}/mathml/decapodes", json=mathml).json()