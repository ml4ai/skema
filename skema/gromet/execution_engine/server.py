from typing import Dict
from tempfile import TemporaryDirectory
from pathlib import Path

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from skema.gromet.execution_engine.execution_engine import ExecutionEngine
HOST = "localhost"
PORT = 7687

router = APIRouter()

class EnrichmentReqest(BaseModel):
    amr: Dict = Field(
        description="The amr to enrich with parameter values",
        example={
            "semantics": {
                "ode":{
                    "parameters":[
                        {"name": "a"},
                        {"name": "b"},
                        {"name": "c"}
                    ]
                }
            }
        }
    )
    source: str = Field(
        description="The raw source code to extract parameter values from",
        example="a=1\nb=a+1\nc=b-a"
    )
    filename: str = Field(
        description="The filename of the file passed in the 'source' field",
        example="source.py"
    )

@router.post("/amr-enrichment", summary="Given an amr and source code, return an enriched amr with parameter values filled in.", response_model=Dict)
def amr_enrichment(request: EnrichmentReqest):
    """
    Endpoint for enriching amr with parameter values.
    ### Python example

    ```
    import requests

    request = {
        "amr": {
            "semantics": {
                "ode":{
                    "parameters":[
                        {"name": "a"},
                        {"name": "b"},
                        {"name": "c"}
                    ]
                }
            }
        },
        "source": "a=1\\nb=a+1\\nc=b-a",
        "filename": "source.py"
    }

    response=client.post("/execution_engine/amr-enrichment", json=request)
    enriched_amr = response.json()
    """
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / request.filename
        source_path.write_text(request.source)

        engine = ExecutionEngine(HOST, PORT, source_path)
        engine.execute(module=True)
        
        return engine.enrich_amr(request.amr)

app = FastAPI()
app.include_router(
    router,
    prefix="/execution_engine",
    tags=["execution_engine"],
)
