# Client code for SKEMA TR
import io
import itertools as it
import json
import tempfile
import time
from pathlib import Path
from typing import List, Union, BinaryIO, Callable
from typing import Optional, Dict, Any
from zipfile import ZipFile

import pandas as pd
import requests
from askem_extractions.data_model import AttributeCollection
from askem_extractions.importers import import_arizona
from fastapi import APIRouter, FastAPI, UploadFile, Response, status

from skema.rest.proxies import SKEMA_TR_ADDRESS, MIT_TR_ADDRESS, OPENAI_KEY, COSMOS_ADDRESS
from skema.rest.schema import (
    TextReadingInputDocuments,
    TextReadingAnnotationsOutput,
    TextReadingDocumentResults,
    TextReadingError, MiraGroundingInputs, MiraGroundingOutputItem, TextReadingEvaluationResults,
)
from skema.rest.utils import compute_text_reading_evaluation

router = APIRouter()


# Utility code for the endpoints

def annotate_with_skema(
        endpoint: str,
        input_: Union[str, List[str], List[Dict], List[List[Dict]]]) -> List[Dict[str, Any]]:
    """ Blueprint for calling the SKEMA-TR API """

    if isinstance(input_, (str, dict)):
        payload = [
            input_
        ]  # If the text to annotate is a single string representing the contents of a document, make it a list with
        # a single element
    else:
        payload = input_  # if the text to annotate is already a list of documents to annotate, it is the payload itself
    response = requests.post(endpoint, json=payload, timeout=600)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(
            f"Calling {endpoint} failed with HTTP code {response.status_code}"
        )


def annotate_text_with_skema(text: Union[str, List[str]]) -> List[Dict[str, Any]]:
    return annotate_with_skema(f"{SKEMA_TR_ADDRESS}/textFileToMentions", text)


def annotate_pdfs_with_skema(
        pdfs: Union[List[List[Dict]], List[Dict]]) -> List[Dict[str, Any]]:
    return annotate_with_skema(f"{SKEMA_TR_ADDRESS}/cosmosJsonToMentions", pdfs)


# Client code for MIT TR
def annotate_text_with_mit(
        texts: Union[str, List[str]]
) -> Union[List[Dict[str, Any]], str]:
    endpoint = f"{MIT_TR_ADDRESS}/annotation/upload_file_extract"
    if isinstance(texts, str):
        texts = [
            texts
        ]  # If the text to annotate is a single string representing the contents of a document, make it a list with
        # a single element

    # TODO parallelize this
    return_values = list()
    for ix, text in enumerate(texts):
        params = {"gpt_key": OPENAI_KEY}
        files = {"file": io.StringIO(text)}
        response = requests.post(endpoint, params=params, files=files)
        try:
            if response.status_code == 200:
                return_values.append(response.json())
            else:
                return_values.append(
                    f"Calling {endpoint} on the {ix}th input failed with HTTP code {response.status_code}"
                )
        except Exception as ex:
            return_values.append(
                f"Calling {endpoint} on the {ix}th input failed with exception {ex}"
            )
    return return_values


def normalize_extractions(
        arizona_extractions: Optional[Dict[str, Any]], mit_extractions: Optional[Dict]
) -> AttributeCollection:
    collections = list()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        skema_path = tmp_dir / "skema.json"

        canonical_mit, canonical_arizona = None, None

        if arizona_extractions:
            try:
                with skema_path.open("w") as f:
                    json.dump(arizona_extractions, f)
                canonical_arizona = import_arizona(Path(skema_path))
                collections.append(canonical_arizona)
            except Exception as ex:
                print(ex)
        if mit_extractions:
            try:
                # MIT extractions already come normalized
                canonical_mit = AttributeCollection.from_json(mit_extractions)
                collections.append(canonical_mit)
            except Exception as ex:
                print(ex)

        if arizona_extractions and mit_extractions:
            # Merge both with some de de-duplications
            params = {"gpt_key": OPENAI_KEY}

            skema_path = tmp_dir / "canonical_skema.json"
            mit_path = tmp_dir / "canonical_mit.json"

            canonical_arizona.save_json(skema_path)
            canonical_mit.save_json(mit_path)

            data = {
                "mit_file": mit_path.open(),
                "arizona_file": skema_path.open(),
            }
            response = requests.post(
                f"{MIT_TR_ADDRESS}/integration/get_mapping", params=params, files=data
            )

            # MIT merges the collection for us
            if response.status_code == 200:
                merged_collection = AttributeCollection.from_json(response.json())
                # Return the merged collection here
                return merged_collection

    # Merge the collections into a attribute collection
    attributes = list(it.chain.from_iterable(c.attributes for c in collections))

    return AttributeCollection(attributes=attributes)


def parquet_to_json(path):
    parquet_df = pd.read_parquet(path)
    parquet_json = parquet_df.to_json()
    parquet_data = json.loads(parquet_json)

    if len(parquet_data) > 0:
        parquet_data_keys = list(parquet_data.keys())
        num_data_rows = max(
            [int(k) for k in parquet_data[parquet_data_keys[0]]]
        )

        row_order_parquet_data = [dict() for i in range(num_data_rows + 1)]
        for field_key, row_data in parquet_data.items():
            for row_idx, datum in row_data.items():
                row_idx_num = int(row_idx)
                row_order_parquet_data[row_idx_num][field_key] = datum

        # if filename == "documents.parquet":
        # Sorts the content sections by page number and then by
        # bounding box location. Use x-pos first to account for
        # multi-column documents and then sort by y-pos.
        row_order_parquet_data.sort(
            key=lambda d: (
                d["page_num"],
                d["bounding_box"][0]
                // 500,  # allows for indentation while still catching items across the center line
                # (d["bounding_box"][0]) // 100
                # + round((d["bounding_box"][0] % 100 // 10) / 10),
                d["bounding_box"][1],
            )
        )

        edits = list()
        for e1, extraction1 in enumerate(row_order_parquet_data):
            (ext1_x1, ext1_y1, ext1_x2, ext1_y2) = extraction1[
                "bounding_box"
            ]
            # Don't bother processing for left-justified or centered
            # content ... only right column content needs to be checked
            if ext1_x1 < 500:
                continue

            ext1_page_num = extraction1["page_num"]
            found_col_break = False
            insertion_index = -1
            t1 = e1
            while t1 > 0:
                extraction2 = row_order_parquet_data[t1 - 1]
                ext2_page_num = extraction2["page_num"]
                # If the previous sorted entry is on an earlier page
                # then we can stop our search
                if ext1_page_num > ext2_page_num:
                    break

                (ext2_x1, ext2_y1, ext2_x2, ext2_y2) = extraction2[
                    "bounding_box"
                ]

                if ext1_y2 <= ext2_y1:
                    ext2_xspan = ext2_x2 - ext2_x1
                    # Useful heuristic cutoff for now
                    if ext2_xspan >= 800:
                        found_col_break = True
                        insertion_index = t1 - 1
                t1 -= 1
            if found_col_break:
                edits.append(
                    {
                        "del_idx": e1,
                        "ins_idx": insertion_index,
                        "val": extraction1,
                    }
                )
        for edit_dict in edits:
            del row_order_parquet_data[edit_dict["del_idx"]]
            row_order_parquet_data.insert(
                edit_dict["ins_idx"], edit_dict["val"]
            )
        row_order_parquet_data.sort(key=lambda d: (d["pdf_name"]))

        name2results = dict()
        for row_data in row_order_parquet_data:
            if row_data["pdf_name"] in name2results:
                name2results[row_data["pdf_name"]].append(row_data)
            else:
                name2results[row_data["pdf_name"]] = [row_data]

        return next(iter(name2results.items()))[1]


def cosmos_client(name: str, data: BinaryIO):
    """
    Posts a pdf to COSMOS and returns the JSON representation of the parquet file

    """

    # Create POST request to COSMOS server
    # Prep the pdf data for upload
    files = [
        ("pdf", (name, data, 'application/pdf')),
    ]
    response = requests.post(f"{COSMOS_ADDRESS}/process/", files=files)

    if response.status_code == status.HTTP_202_ACCEPTED:

        callback_endpoints = response.json()

        for retry_num in range(200):
            time.sleep(3)  # Retry in ten seconds
            poll = requests.get(f"{callback_endpoints['status_endpoint']}")
            if poll.status_code == status.HTTP_200_OK:
                poll_results = poll.json()
                # If the job is completed, fetch the results
                if poll_results['job_completed']:
                    cosmos_response = requests.get(f"{callback_endpoints['result_endpoint']}")
                    if cosmos_response.status_code == status.HTTP_200_OK:
                        data = cosmos_response.content
                        with ZipFile(io.BytesIO(data)) as z:
                            for file in z.namelist():
                                if file.endswith(".parquet") and \
                                        not file.endswith("_figures.parquet") and \
                                        not file.endswith("_pdfs.parquet") and \
                                        not file.endswith("_tables.parquet") and \
                                        not file.endswith("_sections.parquet") and \
                                        not file.endswith("_equations.parquet"):
                                    # convert parquet to json
                                    with z.open(file) as zf:
                                        json_data = parquet_to_json(zf)
                                        return json_data
                        # Shouldn't reach this point
                        raise RuntimeError("COSMOS data doesn't include document file for annotation")

                    else:
                        raise RuntimeError(
                            f"COSMOS Result Error - STATUS CODE: {response.status_code} - {COSMOS_ADDRESS}")
                # If not, just wait until the next iteration
                else:
                    pass

        # If we reached this point, we time out
        raise TimeoutError(f"Timed out waiting for COSMOS on retry num {retry_num + 1}")

    else:
        raise RuntimeError(f"COSMOS Error - STATUS CODE: {response.status_code} - {COSMOS_ADDRESS}")


def merge_pipelines_results(
        skema_extractions,
        mit_extractions,
        general_skema_error,
        general_mit_error,
        annotate_skema,
        annotate_mit):
    """ Merges and de-duplicates text extractions from pipelines"""

    # Build the generalized errors list
    generalized_errors = list()
    if general_skema_error:
        generalized_errors.append(
            TextReadingError(
                pipeline="SKEMA",
                message=general_skema_error
            )
        )
    if general_mit_error:
        generalized_errors.append(
            TextReadingError(
                pipeline="MIT",
                message=general_mit_error
            )
        )

    # Build the results and input-specific errors
    results = list()
    errors = list()
    assert len(skema_extractions) == len(
        mit_extractions
    ), "Both pipeline results lists should have the same length"
    for skema, mit in zip(skema_extractions, mit_extractions):
        if annotate_skema and isinstance(skema, str):
            errors.append(TextReadingError(pipeline="SKEMA", message=skema))
            skema = None

        if annotate_mit and isinstance(mit, str):
            errors.append(TextReadingError(pipeline="MIT", message=mit))
            mit = None

        normalized = normalize_extractions(
            arizona_extractions=skema, mit_extractions=mit
        )
        results.append(
            TextReadingDocumentResults(
                data=normalized if normalized.attributes else None,
                errors=errors if errors else None,
            )
        )

    return TextReadingAnnotationsOutput(
        outputs=results,
        generalized_errors=generalized_errors if generalized_errors else None
    )


def integrated_extractions(
        response: Response,
        skema_annotator: Callable,
        skema_inputs: List[Union[str, List[Dict]]],
        mit_inputs: List[str],
        annotate_skema: bool = True,
        annotate_mit: bool = True,
) -> TextReadingAnnotationsOutput:
    """
    Run both text extractors and merge the results.
    This is the annotation logic shared between different input formats
    """

    # Initialize the extractions to an empty list of arrays
    skema_extractions = [[] for t in skema_inputs]
    mit_extractions = [[] for t in mit_inputs]
    skema_error = None
    mit_error = None

    if annotate_skema:
        try:
            skema_extractions = skema_annotator(skema_inputs)
        except Exception as ex:
            skema_error = f"Problem annotating with SKEMA: {ex}"

    if annotate_mit:
        try:
            mit_extractions = annotate_text_with_mit(mit_inputs)
        except Exception as ex:
            mit_error = f"Problem annotating with MIT: {ex}"

    return_val = merge_pipelines_results(
        skema_extractions,
        mit_extractions,
        skema_error,
        mit_error,
        annotate_skema,
        annotate_mit
    )

    # If there is any error, set the response's status code to 207
    if skema_error or mit_error or any(o.errors is not None for o in return_val.outputs):
        response.status_code = status.HTTP_207_MULTI_STATUS

    return return_val


# End utility code for the endpoints


@router.post(
    "/integrated-text-extractions",
    summary="Posts one or more plain text documents and annotates with SKEMA and/or MIT text reading pipelines",
    status_code=200
)
async def integrated_text_extractions(
        response: Response,
        texts: TextReadingInputDocuments,
        annotate_skema: bool = True,
        annotate_mit: bool = True,
) -> TextReadingAnnotationsOutput:
    """
    ### Python example
    ```
    params = {
       "annotate_skema":True,
       "annotate_mit": True
    }

    files = [("pdfs", ("paper.txt", open("paper.txt", "rb")))]

    response = request.post(f"{URL}/text-reading/integrated-text-extractions", params=params, files=files)
    if response.status_code == 200:
        data = response.json()
    ```
    """
    # Get the input plain texts
    texts = texts.texts

    # Run the text extractors
    return integrated_extractions(
        response,
        annotate_text_with_skema,
        texts,
        texts,
        annotate_skema,
        annotate_mit
    )


@router.post(
    "/integrated-pdf-extractions",
    summary="Posts one or more pdf documents and annotates with SKEMA and/or MIT text reading pipelines",
    status_code=200
)
async def integrated_pdf_extractions(
        response: Response,
        pdfs: List[UploadFile],
        annotate_skema: bool = True,
        annotate_mit: bool = True
) -> TextReadingAnnotationsOutput:
    """

    ### Python example
    ```
    params = {
       "annotate_skema":True,
       "annotate_mit": True
    }

    files = [("pdfs", ("ijerp.pdf", open("ijerp.pdf", "rb")))]

    response = request.post(f"{URL}/text-reading/integrated-pdf-extractions", params=params, files=files)
    if response.status_code == 200:
        data = response.json()
    ```
    """
    # TODO: Make this handle multiple pdf files in parallel
    # Call COSMOS on the pdfs
    cosmos_data = list()
    for pdf in pdfs:
        if pdf.filename.endswith("json"):
            json_data = json.load(pdf.file)
        else:
            json_data = cosmos_client(pdf.filename, pdf.file)
        cosmos_data.append(json_data)

    # Get the plain text version from cosmos, passed through to MIT pipeline
    plain_texts = ['\n'.join(block['content'] for block in c) for c in cosmos_data]

    # Run the text extractors
    return integrated_extractions(
        response,
        annotate_pdfs_with_skema,
        cosmos_data,
        plain_texts,
        annotate_skema,
        annotate_mit
    )


# These are some direct proxies to the SKEMA and MIT APIs
@router.post(
    "/cosmos_to_json",
    status_code=200,
)
async def cosmos_to_json(pdf: UploadFile) -> List[Dict]:
    """ Calls COSMOS on a pdf and converts the data into json

        ### Python example
        ```
        response = requests.post(f"{endpoint}/text-reading/cosmos_to_json",
                        files=[
                            ("pdf", ("ijerp.pdf", open("ijerph-18-09027.pdf", 'rb')))
                        ]
                    )
        ```
    """
    return cosmos_client(pdf.filename, pdf.file)


@router.post(
    "/ground_to_mira",
    status_code=200,
    response_model=List[List[MiraGroundingOutputItem]]
)
async def ground_to_mira(k: int, queries: MiraGroundingInputs, response: Response) -> List[
    List[MiraGroundingOutputItem]]:
    """ Proxy to the MIRA grounding functionality on the SKEMA TR service

        ### Python example
        ```
        queries = {"queries": ["infected", "suceptible"]}
        params = {"k": 5}
        response = requests.post(f"{endpoint}/text-reading/ground_to_mira", params=params, json=queries)

        if response.status_code == 200:
            results = response.json()
        ```
    """
    params = {
        "k": k
    }
    headers = {
        "Content-Type": "text/plain"
    }
    payload = "\n".join(queries.queries)
    inner_response = requests.post(f"{SKEMA_TR_ADDRESS}/groundStringsToMira", headers=headers, params=params,
                                   data=payload)

    response.status_code = inner_response.status_code

    if inner_response.status_code == 200:
        return [[MiraGroundingOutputItem(**o) for o in q] for q in inner_response.json()]
    else:
        return inner_response.content


@router.post("/cards/get_model_card")
async def get_model_card(text_file: UploadFile, code_file: UploadFile, response: Response):
    """ Calls the model card endpoint from MIT's pipeline

        ### Python example
        ```
        files = {
            "text_file": ('text_file.txt", open("text_file.txt", 'rb')),
            "code_file": ('code_file.py", open("code_file.py", 'rb')),
        }

        response = requests.post(f"{endpoint}/text-reading/cards/get_model_card", files=files)
        ```
    """

    params = {
        "gpt_key": OPENAI_KEY,
    }
    files = {
        "text_file": (text_file.filename, text_file.file, "text/plain"),
        "code_file": (code_file.filename, code_file.file, "text/plain")
    }

    inner_response = requests.post(f"{MIT_TR_ADDRESS}/cards/get_model_card", params=params, files=files)

    response.status_code = inner_response.status_code
    return inner_response.json()


@router.post("/cards/get_data_card")
async def get_data_card(smart: bool, csv_file: UploadFile, doc_file: UploadFile, response: Response):
    """
        Calls the data card endpoint from MIT's pipeline.
        Smart run provides better results but may result in slow response times as a consequence of extra GPT calls.

        ### Python example
        ```
        params = {
            "smart": False
        }

        files = {
            "csv_file": ('csv_file.csv", open("csv_file.csv", 'rb')),
            "doc_file": ('doc_file.txt", open("doc_file.txt", 'rb')),
        }

        response = requests.post(f"{endpoint}/text-reading/cards/get_data_card", params=params files=files)
        ```
    """

    params = {
        "gpt_key": OPENAI_KEY,
        "smart": smart
    }
    files = {
        "csv_file": (csv_file.filename, csv_file.file, "text/csv"),
        "doc_file": (doc_file.filename, doc_file.file, "text/plain")
    }

    inner_response = requests.post(f"{MIT_TR_ADDRESS}/cards/get_data_card", params=params, files=files)

    response.status_code = inner_response.status_code
    return inner_response.json()


####


@router.get(
    "/healthcheck",
    summary="Check health of integrated text reading service",
    response_model=int,
    status_code=200,
    responses={
        200: {
            "model": int,
            "description": "All component services are healthy (200 status)",
        },
        500: {
            "model": int,
            "description": "Internal error occurred",
            "example_value": 500
        },
        502: {
            "model": int,
            "description": "Either text reading service is not available"
        }

    },
)
def healthcheck() -> int:
    # SKEMA health check
    skema_endpoint = f"{SKEMA_TR_ADDRESS}/api/skema"
    try:
        skema_response = requests.get(skema_endpoint, timeout=10)
    except Exception:
        return status.HTTP_500_INTERNAL_SERVER_ERROR

    # TODO replace this with a proper healthcheck endpoint
    mit_endpoint = f"{MIT_TR_ADDRESS}/annotation/find_text_vars/"
    mit_params = {"gpt_key": OPENAI_KEY}
    files = {"file": io.StringIO("x = 0")}
    try:
        mit_response = requests.post(mit_endpoint, params=mit_params, files=files, timeout=10)
    except Exception:
        return status.HTTP_502_BAD_GATEWAY
    ######################################################

    status_code = (
        status.HTTP_200_OK
        if all(resp.status_code == 200 for resp in [skema_response, mit_response])
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return status_code


@router.get("/eval", response_model=TextReadingEvaluationResults, status_code=200)
def quantitative_eval() -> TextReadingEvaluationResults:
    """ Compares the SIDARTHE paper extractions against ground truth extractions """

    # Read ground truth annotations
    with (Path(__file__).parents[0] / "data" / "sidarthe_annotations.json").open() as f:
        gt_data = json.load(f)

    # Read the SKEMA extractions
    extractions = AttributeCollection.from_json(Path(__file__).parents[0] / "data" / "extractions_sidarthe_skema.json")

    return compute_text_reading_evaluation(gt_data, extractions)


@router.post("/eval", response_model=TextReadingEvaluationResults, status_code=200)
def quantitative_eval(extractions_file: UploadFile, gt_annotations: UploadFile):
    """
    # Gets performance metrics of a set of text extractions againts a ground truth annotations file.

    ## Example:
    ```python
    files = {
        "extractions_file": ("paper_variable_extractions.json", open("paper_variable_extractions.json", 'rb')),
        "gt_annotations": ("paper_gt_annotations.json", open("paper_gt_annotations.json", 'rb')),
    }

    response = requests.post(f"{endpoint}/text-reading/eval", files=files)
    ```

    """

    gt_data = json.load(gt_annotations.file)

    # Support both Attribute Collections serialized and within the envelop of this rest API
    extractions_json = json.load(extractions_file.file)
    try:
        extractions = AttributeCollection.from_json(extractions_json)
    except KeyError:
        extractions_file.file.seek(0)
        service_output = json.load(extractions_file.file)
        collections = list()
        for collection in service_output['outputs']:
            collection = AttributeCollection.from_json(collection['data'])
            collections.append(collection)

        extractions = AttributeCollection(
            attributes=list(it.chain.from_iterable(c.attributes for c in collections)))

    return compute_text_reading_evaluation(gt_data, extractions)


app = FastAPI()
app.include_router(router)
