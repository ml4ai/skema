import json
import itertools as it

from askem_extractions.data_model import AttributeCollection
from fastapi import UploadFile, File, APIRouter, FastAPI
from pydantic import Json


from skema.metal.model_linker.skema_model_linker.linkers import PetriNetLinker, RegNetLinker
from skema.metal.model_linker.skema_model_linker.link_amr import replace_xml_codepoints
from skema.rest.schema import TextReadingAnnotationsOutput

router = APIRouter()


@router.post(
    "/link_amr",
)
def link_amr(amr_type: str,
             similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
             similarity_threshold: float = 0.5,
             amr_file: UploadFile = File(...),
             text_extractions_file: UploadFile = File(...)):
    """ Links an AMR to a text extractions file

        ### Python example
        ```
        params = {
          "amr_type": "petrinet"
        }

        files = {
          "amr_file": ("amr.json", open("amr.json"), "application/json"),
          "text_extractions_file": ("extractions.json", open("extractions.json"), "application/json")
        }

        response = requests.post(f"{ENDPOINT}/metal/link_amr", params=params, files=files)
        if response.status_code == 200:
            enriched_amr = response.json()
        ```
    """

    # Load the AMR
    amr = json.load(amr_file.file)
    amr = replace_xml_codepoints(amr)

    # Load the extractions, that come out of the TR Proxy endpoint
    raw_extractions = json.load(text_extractions_file.file)
    text_extractions = [AttributeCollection.from_json(o['data']) for o in raw_extractions['outputs']]
    # text_extractions = TextReadingAnnotationsOutput(**json.load(text_extractions_file.file))

    # Merge all the attribute collections
    extractions = AttributeCollection(
        attributes=list(
            it.chain.from_iterable(o.attributes for o in text_extractions)
        )
    )

    # Link the AMR
    if amr_type == "petrinet":
        Linker = PetriNetLinker
    elif amr_type == "regnet":
        Linker = RegNetLinker
    else:
        raise NotImplementedError(f"{amr_type} AMR currently not supported")

    linker = Linker(model_name=similarity_model, sim_threshold=similarity_threshold)

    return linker.link_model_to_text_extractions(amr, extractions)


@router.get(
    "/healthcheck",
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
        }
    },
)
def healthcheck():
    return 200


app = FastAPI()
app.include_router(router)
