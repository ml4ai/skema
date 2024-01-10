# -*- coding: utf-8 -*-

from fastapi import Depends, FastAPI, APIRouter, status
from skema.isa.lib import align_mathml_eqs
import skema.isa.data as isa_data
from skema.rest import utils
from pydantic import BaseModel
import httpx

from skema.rest.proxies import SKEMA_RS_ADDESS

router = APIRouter()


# Model for ISA_Result
class ISA_Result(BaseModel):
    matching_ratio: float = None
    union_graph: str = None


@router.get(
    "/healthcheck", 
    summary="Status of ISA service",
    response_model=int,
    status_code=status.HTTP_200_OK
)
async def healthcheck(client: httpx.AsyncClient = Depends(utils.get_client)) -> int:
    res = await client.get(f"{SKEMA_RS_ADDESS}/ping")
    return res.status_code


@router.post(
    "/align-eqns", 
    summary="Align two MathML equations"
)
async def align_eqns(
    mml1: str, mml2: str, mention_json1: str = "", mention_json2: str = ""
) -> ISA_Result:
    f"""
    Endpoint for align two MathML equations.

    ### Python example

    ```
    import requests

    request = {{
        "mml1": {isa_data.mml},
        "mml2": {isa_data.mml}
    }}

    response=requests.post("/isa/align-eqns", json=request)
    res = response.json()
    """
    (
        matching_ratio,
        num_diff_edges,
        node_labels1,
        node_labels2,
        aligned_indices1,
        aligned_indices2,
        union_graph,
        perfectly_matched_indices1,
    ) = align_mathml_eqs(mml1, mml2, mention_json1, mention_json2)
    return ISA_Result(
      matching_ratio = matching_ratio,
      union_graph = union_graph.to_string()
    )


app = FastAPI()
app.include_router(
    router,
    prefix="/isa",
    tags=["isa"],
)
