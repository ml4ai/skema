# -*- coding: utf-8 -*-
"""
Proxies requests to skema_rs API to provide a unified API.
"""


from typing import Any, Dict, List, Text
from skema.rest.proxies import SKEMA_RS_ADDESS
from fastapi import APIRouter, Depends
from skema.rest import utils
# TODO: replace use of requests with httpx
import httpx


router = APIRouter()


# FIXME: make GrometFunctionModuleCollection a pydantic model via code gen
@router.post("/model", summary="Pushes gromet (function network) to the graph database", include_in_schema=False)
async def post_model(gromet: Dict[Text, Any], client: httpx.AsyncClient = Depends(utils.get_client)):
    res = await client.post(f"{SKEMA_RS_ADDESS}/models", json=gromet)
    return res.json()


@router.get("/models", summary="Gets function network IDs from the graph database")
async def get_models(client: httpx.AsyncClient = Depends(utils.get_client)) -> List[int]:
    res = await client.get(f"{SKEMA_RS_ADDESS}/models")
    print(f"request: {res}")
    return res.json()


@router.get("/ping", summary="Status of MORAE service")
async def healthcheck(client: httpx.AsyncClient = Depends(utils.get_client)) -> int:
    res = await client.get(f"{SKEMA_RS_ADDESS}/ping")
    return res.status_code


@router.get("/version", summary="Status of MORAE service")
async def versioncheck(client: httpx.AsyncClient = Depends(utils.get_client)) -> str:
    res = await client.get(f"{SKEMA_RS_ADDESS}/version")
    return res.text

@router.post("/mathml/decapodes", summary="Gets Decapodes from a list of MathML strings")
async def get_decapodes(mathml: List[str], client: httpx.AsyncClient = Depends(utils.get_client)) -> Dict[Text, Any]:
    res = await client.put(f"{SKEMA_RS_ADDESS}/mathml/decapodes", json=mathml)
    return res.json()