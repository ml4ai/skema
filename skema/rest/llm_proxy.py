from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)
from fastapi import APIRouter, FastAPI, File, UploadFile
from io import BytesIO
from zipfile import ZipFile
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from skema.rest.proxies import SKEMA_OPENAI_KEY

router = APIRouter()

class Dynamics(BaseModel):
    """
    Dynamics Data Model for capturing dynamics within a CodeFile.
    """

    name: Optional[str] = Field(description="Name of the dynamics section.")
    description: Optional[str] = Field(description="Description of the dynamics.")
    block: List[str] = Field(
        description="A list containing strings indicating the line numbers in the file that contain the dynamics, e.g., ['L205-L213', 'L225-L230']."
    )

@router.post(
    "/linespan-given-filepaths-zip",
    summary=(
        "Send a zip file containing a code file,"
        " get a line span of the dynamics back."
    ),
)
async def get_lines_of_model(zip_file: UploadFile = File()) -> LineSpan:
    """
    Endpoint for generating a line span containing the dynamics from a zip archive. Currently
    it only expects there to be one python file in the zip. There can be other files, such as a
    README.md, but only one .py. Future versions will generalize support to arbritary zip contents. 

    ### Python example
    ```
    import requests

    files = {
      "zip_file": open(zip_path, "rb"),
    }

    response = requests.post(f"{ENDPOINT}/morae/linespan-given-filepaths-zip", files=files)
    gromet_json = response.json()
    """
    files=[]
    blobs=[]
    block=[]
    with ZipFile(BytesIO(zip_file.file.read()), "r") as zip:
        for file in zip.namelist():
            file_obj = Path(file)
            if file_obj.suffix in [".py"]:
                files.append(file)
                blobs.append(zip.open(file).read())

    # read in the code, for the prompt
    code = blobs[0].decode("utf-8") # needs to be regular string, not byte string
    file = files[0]
    # json for the fn construction
    single_snippet_payload = {
            "files": [file],
            "blobs": [code],
        }

    # this is the formatting instructions
    response_schemas = [
        ResponseSchema(name="model_function", description="The name of the function that contains the model dynamics")
    ]

    # for structured output parsing, converts schema to langhchain object
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # for structured output parsing, makes the instructions to be passed as a variable to prompt template
    format_instructions = output_parser.get_format_instructions()

    # low temp as is not generative
    temperature = 0.1

    # initialize the models
    openai = ChatOpenAI(
        temperature=temperature,
        model_name='gpt-3.5-turbo',
        openai_api_key=SKEMA_OPENAI_KEY
    )

    # construct the prompts
    template="You are a assistant that answers questions about code."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="Find the function that contains the model dynamics in {code} \n{format_instructions}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # combining the templates for a chat template
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # formatting the prompt with input variables
    formatted_prompt = chat_prompt.format_prompt(code=code, format_instructions = format_instructions).to_messages()

    # running the model
    output = openai(formatted_prompt)

    # parsing the output
    try:
        parsed_output = output_parser.parse(output.content)

        function_name = parsed_output['model_function']

        # Get the FN from it
        url = "https://api.askem.lum.ai/code2fn/fn-given-filepaths"
        response_zip = requests.post(url, json=single_snippet_payload)

        # get metadata entry for function
        for entry in response_zip.json()['modules'][0]['fn_array']:
            try:
                if entry['b'][0]['name'][0:len(function_name)] == function_name:
                    metadata_idx = entry['b'][0]['metadata']
            except:
                None

        # get line span using metadata
        for (i,metadata) in enumerate(response_zip.json()['modules'][0]['metadata_collection']):
            if i == (metadata_idx - 1):
                line_begin = metadata[0]['line_begin']
                line_end =  metadata[0]['line_end']
    except:
        print("Failed to parse dynamics")
        line_begin = 0
        line_end = 0

    block.append(f"L{line_begin}-L{line_end}")

    output = Dynamics(name=None, description=None, block=block)
    return output


app = FastAPI()
app.include_router(router)