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
    amr: Dict = Field(),
    source: str = Field(),
    filename: str = Field()

@router.post("/amr-enrichment", summary="Ping endpoint to test health of service")
def amr_enrichment(request: EnrichmentReqest):
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
