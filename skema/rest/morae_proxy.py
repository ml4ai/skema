# -*- coding: utf-8 -*-
"""
Proxies requests to skema_rs API to provide a unified API.
"""

from enum import Enum

from typing import Any, Dict, List, Text, Union
from typing_extensions import Annotated
from skema.rest.proxies import SKEMA_RS_ADDESS
from fastapi import APIRouter, FastAPI, Body, File, Response, Request, Query
from pydantic import BaseModel, Field
import requests


router = APIRouter()

# FIXME: make GrometFunctionModuleCollection a pydantic model via code gen
@router.post("/model", summary="Pushes gromet (function network) to the graph database")
async def post_model(gromet: Dict[Text, Any]):
    return requests.post(f"{SKEMA_RS_ADDESS}/models", json=gromet).json()

@router.get("/models", summary="Gets function network IDs from the graph database")
async def get_models() -> List[str]:
    return requests.get(f"{SKEMA_RS_ADDESS}/models").json()

@router.get("/ping", summary="Status of MORAE service")
async def healthcheck() -> int:
    return requests.get(f"{SKEMA_RS_ADDESS}/ping").status_code