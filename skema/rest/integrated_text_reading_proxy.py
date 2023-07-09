# Client code for SKEMA TR
import itertools as it
import json
import os
import tempfile
from pathlib import Path
from typing import List, Union, Text
from typing import Optional, Dict, Any

import requests
from askem_extractions.data_model import AttributeCollection
from askem_extractions.importers import import_arizona, import_mit
from askem_extractions.importers.mit import merge_collections
from fastapi import APIRouter, FastAPI
from starlette import status

from skema.rest.proxies import SKEMA_TR_ADDRESS, MIT_TR_ADDRESS, OPENAI_KEY
from skema.rest.schema import TextReadingInputDocuments, TextReadingAnnotationsOutput, TextReadingDocumentResults, \
    TextReadingError

router = APIRouter()


# Utility code for the endpoints
def annotate_text_with_skema(text: Union[str, List[str]]) -> List[Dict[str, Any]]:
    endpoint = f"{SKEMA_TR_ADDRESS}/textFileToMentions"
    if isinstance(text, str):
        payload = [
            text
        ]  # If the text to annotate is a single string representing the contents of a document, make it a list with a single element
    else:
        payload = text  # if the text to annotate is already a list of documents to annotate, it is the payload itself
    response = requests.post(endpoint, json=payload, timeout=600)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(
            f"Calling {endpoint} failed with HTTP code {response.status_code}"
        )


# Client code for MIT TR


def annotate_text_with_mit(texts: Union[str, List[str]]) -> Union[List[Dict[str, Any]], str]:
    endpoint = f"{MIT_TR_ADDRESS}/annotation/find_text_vars/"
    if isinstance(texts, str):
        texts = [
            texts
        ]  # If the text to annotate is a single string representing the contents of a document, make it a list with a single element

    # TODO paralelize this
    return_values = list()
    for ix, text in enumerate(texts):
        params = {"gpt_key": OPENAI_KEY, "text": text}
        response = requests.post(endpoint, params=params)
        if response.status_code == 200:
            return_values.append(response.json())
        else:
            return_values.append(
                f"Calling {endpoint} on the {ix}th input failed with HTTP code {response.status_code}"
            )
    return return_values


def normalize_extractions(
    arizona_extractions: Optional[Dict[str, Any]], mit_extractions: Optional[Dict]
) -> AttributeCollection:
    collections = list()
    with tempfile.TemporaryDirectory() as tmpdirname:
        skema_path = os.path.join(tmpdirname, "skema.json")
        mit_path = os.path.join(tmpdirname, "mit.json")

        if arizona_extractions:
            try:
                with open(skema_path, "w") as f:
                    json.dump(arizona_extractions, f)
                canonical_arizona = import_arizona(Path(skema_path))
                collections.append(canonical_arizona)
            except Exception as ex:
                print(ex)
        if mit_extractions:
            try:
                with open(mit_path, "w") as f:
                    json.dump(mit_extractions, f)
                canonical_mit = import_mit(Path(mit_path))
                collections.append(canonical_mit)
            except Exception as ex:
                print(ex)

        if arizona_extractions and mit_extractions:
            # Merge both with some de de-duplications
            params = {"gpt_key": OPENAI_KEY}

            data = {
                "mit_file": json.dumps(AttributeCollection(attributes=canonical_mit).json()),
                "arizona_file": json.dumps(AttributeCollection(attributes=canonical_arizona).json()),
            }
            response = requests.post(
                f"{MIT_TR_ADDRESS}/integration/get_mapping", params=params, data=data
            )

            if response.status_code == 200:
                map_data = response.text
                map_path = os.path.join(tmpdirname, "mapping.txt")
                with open(map_path, "w") as f:
                    f.write(map_data)
                merged_collection = merge_collections(
                    a_collection=collections[0],
                    m_collection=collections[1],
                    map_path=Path(map_path),
                )

                # Return the merged collection here
                return merged_collection

    # Merge the collections into a attribute collection
    attributes = list(it.chain.from_iterable(c.attributes for c in collections))

    return AttributeCollection(attributes=attributes)


# End utility code for the endpoints


@router.post(
    "/integrated-text-extractions",
    summary="Posts one or more plain text documents and annotates with SKEMA and/or MIT text reading pipelines",
)
async def integrated_text_extractions(
    texts: TextReadingInputDocuments,
    annotate_skema: bool = True,
    annotate_mit: bool = True,
) -> TextReadingAnnotationsOutput:
    texts = texts.texts
    skema_extractions = [[] for t in texts]

    if annotate_skema:
        try:
            skema_extractions = annotate_text_with_skema(texts)
        except Exception as ex:
            print(f"Problem annotating with Skema: {ex}")

    mit_extractions = [[] for t in texts]

    if annotate_mit:
        mit_extractions = annotate_text_with_mit(texts)


    results = list()
    errors = list()
    assert len(skema_extractions) == len(mit_extractions), "Both pipeline results lists should have the same length"
    for skema, mit in zip(skema_extractions, mit_extractions):
        if annotate_skema and isinstance(skema, str):
            errors.append(TextReadingError(
                pipeline="SKEMA",
                message=skema
            ))

        if annotate_mit and isinstance(mit, str):
            errors.append(TextReadingError(
                pipeline="MIT",
                message=mit
            ))

        normalized = normalize_extractions(
            arizona_extractions=skema, mit_extractions=mit
        )
        results.append(
            TextReadingDocumentResults(
                data = normalized if normalized.attributes else None,
                errors = errors if errors else None
            )
        )
    return TextReadingAnnotationsOutput(outputs=results)


@router.get(
    "/healthcheck",
    summary="Check health of integrated text reading service",
    response_model=int,
    status_code=200,
)
def healthcheck() -> int:
    # SKEMA health check
    skema_endpoint = f"{SKEMA_TR_ADDRESS}/api/skema"
    skema_response = requests.get(skema_endpoint, timeout=10)

    # TODO replace this with a proper healthcheck endpoint
    mit_endpoint = f"{MIT_TR_ADDRESS}/annotation/find_text_vars/"
    mit_params = {"gpt_key": OPENAI_KEY, "text": "x = 0"}
    mit_response = requests.post(mit_endpoint, params=mit_params, timeout=10)
    ######################################################

    status_code = (
        status.HTTP_200_OK
        if all(
            code == 200
            for code in [skema_response, mit_response]
        )
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return status_code

app = FastAPI()
app.include_router(router)
