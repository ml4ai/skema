import os
import requests

from fastapi import FastAPI, Body

import acsets, petris

app = FastAPI()

@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The service is running."

@app.post(
    "/endpoint1",
    summary=(
        "endpoint 1"
    ),
)
async def root(model_id: str = Body(...)):
    # Determine Memgraph location from 
    skema_server_host = os.getenv("SKEMA_HOST")
    skema_server_port = os.getenv("SKEMA_PORT")
    skema_url = f"http://{skema_server_host}:{skema_server_port}"

    opi_url = f"{skema_url}/models/{model_id}/named_opis"
    opo_url = f"{skema_url}/models/{model_id}/named_opos"
    
    opi_json = requests.get(opi_url).json()
    opo_json = requests.get(opo_url).json()
    
    return {"opis": opi_json, "opos": opo_json}


@app.post(
    "/endpoint2",
    summary=(
        "endpoint2"
    ),
)
async def root(opis: list, opos: list):
    sir = petris.Petri()
    sir.add_species(len(opos))
    trans = petris.Transition
    sir.add_parts(trans, len(opis))
    
    for i, tran in enumerate(opis):
        sir.set_subpart(i, petris.attr_tname, opis[i])

    for j, spec in enumerate(opos):
        sir.set_subpart(j, petris.attr_sname, opos[j])

    return sir.write_json()
   
    
