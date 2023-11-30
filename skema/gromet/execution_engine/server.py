from typing import Dict
from tempfile import TemporaryDirectory
from pathlib import Path

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from skema.rest.proxies import SKEMA_GRAPH_DB_HOST, SKEMA_GRAPH_DB_PORT, SKEMA_GRAPH_DB_PROTO
from skema.gromet.execution_engine.execution_engine import ExecutionEngine
HOST = SKEMA_GRAPH_DB_HOST
PORT = int(SKEMA_GRAPH_DB_PORT)
PROTOCOL = SKEMA_GRAPH_DB_PROTO
print("LOGGING: Lanuching execution engine REST service")
print(f"LOGGING: SKEMA_GRAPH_DB_PROTOCOL {PROTOCOL}")
print(f"LOGGING: SKEMA_GRAPH_DB_HOST {HOST}")
print(f"LOGGING: SKEMA_GRAPH_DB_PORT {PORT}")

router = APIRouter()

class EnrichmentReqest(BaseModel):
    amr: Dict = Field(
        description="The amr to enrich with parameter values",
        examples=[{
            "semantics": {
                "ode":{
                    "parameters":[
                        {"name": "a"},
                        {"name": "b"},
                        {"name": "c"}
                    ]
                }
            }
        }]
    ),
    source: str = Field(
        description="The raw source code to extract parameter values from",
        examples=["a=1\nb=a+1\nc=b-a"]
    ),
    filename: str = Field(
        description="The filename of the file passed in the 'source' field",
        examples=["source.py"]
    )

@router.post("/amr-enrichment", summary="Given an amr and source code, return an enriched amr with parameter values filled in.")
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

        engine = ExecutionEngine(PROTOCOL, HOST, PORT, str(source_path))
        engine.execute(module=True)
        
        return engine.enrich_amr(request.amr)

app = FastAPI()
app.include_router(
    router,
    prefix="/execution-engine",
    tags=["execution-engine"],
)
