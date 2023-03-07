# -*- coding: utf-8 -*-

from fastapi import FastAPI, File
from lib import align_mathml_eqs

# Create a web app using FastAPI

app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The ISA service is running."


@app.put("/align-eqns", summary="Align two MathML equations")
async def align_eqns(file1: str, file2: str):
    """
    Endpoint for align two MathML equations.
    """
    matching_ratio, num_diff_edges, node_labels1, node_labels2, aligned_indices1, aligned_indices2 = align_mathml_eqs(
        file1, file2)
    return matching_ratio