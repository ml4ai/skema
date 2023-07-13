from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
MATHPIX_ANNOTATION_DATA_PATH = "./data/mathpix_ocr_tweaker_data.json"
MODEL_RESULTS_DATA_PATH = "./data/ocr_verify_data.json"


@app.get("/model_results")
async def get_model_results_data(request: Request):
    # Grab data
    with open(MODEL_RESULTS_DATA_PATH, "r") as f:
        model_results = json.load(f)
    max_length = len(model_results)

    # grab draw and search parameters
    draw, search = (
        int(request.query_params["draw"]),
        request.query_params["search[value]"],
    )
    search = search.strip()
    shouldSearch = search != ""

    if not shouldSearch:
        # We don't need to search, so return the requested
        # page of data
        # Ref: https://datatables.net/manual/server-side
        start, length = (
            int(request.query_params["start"]),
            int(request.query_params["length"]),
        )
        length = length if length != -1 else max_length
        result_subset = model_results[start : start + length]
    else:
        # Otherwise search for the closest matches
        result_subset = []
        for record in model_results:
            if search in record["id"]:
                result_subset.append(record)

    return {
        "draw": draw,
        "recordsTotal": max_length,
        "recordsFiltered": max_length if not shouldSearch else len(result_subset),
        "data": result_subset,
    }


@app.get("/mathpix_annotation_data")
async def get_mathpix_annotation_data():
    with open(MATHPIX_ANNOTATION_DATA_PATH, "r") as f:
        annotation_data = json.load(f)
    return annotation_data


@app.post("/mathpix_annotation_data")
async def post_mathpix_annotation_data(request: Request):
    updated_annotation_data = await request.json()

    with open(MATHPIX_ANNOTATION_DATA_PATH, "w") as f:
        json.dump(updated_annotation_data, f, indent=4)

    return {"message": "success"}


@app.get("/", response_class=FileResponse)
async def main():
    return "./static/index.html"
