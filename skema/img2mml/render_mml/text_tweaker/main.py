from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
ANNOTATION_DATA_PATH = "./text_tweaker_annotation_data.json"


@app.get("/annotation_data")
async def get_annotation_data():
    with open(ANNOTATION_DATA_PATH, "r") as f:
        annotation_data = json.load(f)
    return annotation_data


@app.post("/annotation_data")
async def root(request: Request):
    updated_annotation_data = await request.json()

    with open(ANNOTATION_DATA_PATH, "w") as f:
        json.dump(updated_annotation_data, f, indent=4)

    return {"message": "success"}


@app.get("/", response_class=FileResponse)
async def main():
    return "./static/index.html"
