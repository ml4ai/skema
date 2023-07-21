from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import lxml.etree as etree
import json
import os


PARENT_DIRECTORY = os.path.split(os.path.abspath(__file__))[0]
STATIC_FILES_DIRECTORY = os.path.join(PARENT_DIRECTORY, "static/")
MATHPIX_ANNOTATION_DATA_PATH = os.path.join(
    PARENT_DIRECTORY, "data/mathpix_ocr_tweaker_data.json"
)
MODEL_RESULTS_DATA_PATH = os.path.join(PARENT_DIRECTORY, "data/ocr_verify_data.json")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIRECTORY), name="static")


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
    should_search = search != ""
    search_exact = "search_exact" in request.query_params

    if not should_search:
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
            if not search_exact:
                if search in record["id"]:
                    result_subset.append(record)
            else:
                if search == record["id"]:
                    result_subset.append(record)
                    break

    return {
        "draw": draw,
        "recordsTotal": max_length,
        "recordsFiltered": max_length if not should_search else len(result_subset),
        "data": result_subset,
    }


@app.get("/mathpix_annotation_data")
async def get_mathpix_annotation_data(index=-1, id=None):
    index = int(index)

    # Get data
    with open(MATHPIX_ANNOTATION_DATA_PATH, "r") as f:
        annotation_data = json.load(f)

    # Return all records or requested record (search by number or id)
    response = {"data": annotation_data}
    totalRecords = len(annotation_data)

    if id:
        response = {"error", f"Image {id} not found"}
        for i, entry in enumerate(annotation_data):
            if entry["id"] == id:
                index = i
                response = {"data": [entry]}
                break
    elif index != -1:
        response = {"data": [annotation_data[index]]}

    response["totalRecords"] = totalRecords
    response["index"] = index
    return response


@app.post("/mathpix_annotation_data")
async def post_mathpix_annotation_data(request: Request):
    request_body = await request.json()

    index = request_body["index"]
    data = request_body["data"]

    # Update data
    with open(MATHPIX_ANNOTATION_DATA_PATH, "r") as f:
        annotation_data = json.load(f)
    annotation_data[index] = data

    with open(MATHPIX_ANNOTATION_DATA_PATH, "w") as f:
        json.dump(annotation_data, f, indent=4)

    return True


@app.post("/model_results")
async def post_model_results(request: Request):
    request_body = await request.json()

    id, raw_mathml = request_body["id"], request_body["mathml"]

    # Try to pretty print MathML before saving
    mathml = raw_mathml
    try:
        mathml_tree = etree.fromstring(raw_mathml)
        pretty_printed_mathml = etree.tostring(
            mathml_tree, pretty_print=True, encoding="unicode"
        )
        mathml = pretty_printed_mathml
    except:
        print("Warning: Could not pretty print MathML before saving.")

    # Grab data
    with open(MODEL_RESULTS_DATA_PATH, "r") as f:
        model_results = json.load(f)

    # Update and save data
    for result in model_results:
        if result["id"] == id:
            result["mathml"] = mathml
            break

    with open(MODEL_RESULTS_DATA_PATH, "w") as f:
        json.dump(model_results, f, indent=4)

    return True


@app.get("/", response_class=FileResponse)
async def main():
    return "./static/index.html"
