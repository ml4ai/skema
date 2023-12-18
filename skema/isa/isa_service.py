# -*- coding: utf-8 -*-

from fastapi import FastAPI, APIRouter
from skema.isa.lib import align_mathml_eqs
from pydantic import BaseModel


router = APIRouter()


# Model for ISA_Result
class ISA_Result(BaseModel):
    matching_ratio: float = None
    union_graph: str = None


@router.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The ISA service is running."


@router.put("/align-eqns", summary="Align two MathML equations")
async def align_eqns(
    file1: str, file2: str, mention_json1: str = "", mention_json2: str = ""
) -> ISA_Result:
    """
    Endpoint for align two MathML equations.
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
    ) = align_mathml_eqs(file1, file2, mention_json1, mention_json2)
    ir = ISA_Result()
    ir.matching_ratio = matching_ratio
    ir.union_graph = union_graph.to_string()
    return ir


app = FastAPI()
app.include_router(router)
