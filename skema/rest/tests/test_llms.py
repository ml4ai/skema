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
from skema.rest.proxies import SKEMA_OPENAI_KEY

def test_prompt_construction():
    """Tests prompt template instantiation"""
    # TODO: your assertion here that the template instantiation returns a string/valid type

    code = "def sir(\n    s: float, i: float, r: float, beta: float, gamma: float, n: float\n) -> Tuple[float, float, float]:\n    \"\"\"The SIR model, one time step.\"\"\"\n    s_n = (-beta * s * i) + s\n    i_n = (beta * s * i - gamma * i) + i\n    r_n = gamma * i + r\n    scale = n / (s_n + i_n + r_n)\n    return s_n * scale, i_n * scale, r_n * scale"

    # this is the formatting instructions
    response_schemas = [
        ResponseSchema(name="model_function", description="The name of the function that contains the model dynamics")
    ]

    # for structured output parsing, converts schema to langhchain object
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # for structured output parsing, makes the instructions to be passed as a variable to prompt template
    format_instructions = output_parser.get_format_instructions()

    # low temp as is not generative
    temperature = 0.0

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

    parsed_output = output_parser.parse(output.content)

    assert isinstance(parsed_output['model_function'], str)