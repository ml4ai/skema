import json, html
from pathlib import Path
from typing import Optional

import fire.fire_test
from askem_extractions.data_model import AttributeCollection

from .linkers import PetriNetLinker, RegNetLinker

def replace_xml_codepoints(json):
    """ Looks for xml special characters and substitutes them with their unicode character """
    def clean(text):
        return html.unescape(text) if text.startswith("&#") else text

    if isinstance(json, list):
        return [replace_xml_codepoints(elem) for elem in json]
    elif isinstance(json, dict):
        return {clean(k):replace_xml_codepoints(v) for k, v in json.items()}
    elif isinstance(json, str):
        return clean(json)
    else:
        return json

def link_amr(
        amr_path: str,  # Path of the AMR model
        attribute_collection: str,  # Path to the attribute collection
        amr_type: str,  # AMR model type. I.e. "petrinet" or "regnet"
        output_path: Optional[str] = None,  # Output file path
        clean_xml_codepoints: Optional[bool] = False, # Replaces html codepoints with the unicode character
        similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # Transformer model to compute similarities
        similarity_threshold: float = 0.7,  # Cosine similarity threshold for linking
        device: Optional[str] = None  # PyTorch device to run the model on
):
    """ Links and AMR model to an attribute collections from ASKEM text reading pipelines """

    if amr_type == "petrinet":
        Linker = PetriNetLinker
    elif amr_type == "regnet":
        Linker = RegNetLinker
    else:
        raise NotImplementedError(f"{amr_type} AMR currently not supported")

    with open(amr_path) as f:
        amr = json.load(f)
        if clean_xml_codepoints:
            amr = replace_xml_codepoints(amr)

    extractions = AttributeCollection.from_json(attribute_collection)

    linker = Linker(model_name=similarity_model, device=device, sim_threshold=similarity_threshold)

    linked_model = linker.link_model_to_text_extractions(amr, extractions)

    if not output_path:
        input_amr_name = str(Path(amr_path).name)
        output_path = f'linked_{input_amr_name}'

    with open(output_path, 'w') as f:
        json.dump(linked_model, f, default=str, indent=2, ensure_ascii=False)


def main():
    """ Module's entry point"""
    fire.Fire(link_amr)


if __name__ == "__main__":
    main()
