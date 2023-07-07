# This is a sample Python script.
import json
from pathlib import Path
from typing import Optional

import fire.fire_test
from askem_extractions.data_model import AttributeCollection
from skema.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet

from .linkers import GrometLinker


def link_gromet(
        gromet_path: str,  # Path of the AMR model
        attribute_collection: str,  # Path to the attribute collection
        output_path: Optional[str] = None,  # Output file path
        similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # Transformer model to compute similarities
        similarity_threshold: float = 0.7,  # Cosine similarity threshold for linking
        device: Optional[str] = None  # PyTorch device to run the model on
):
    """ Links a Gromet model to an attribute collections from ASKEM text reading pipelines """


    gromet = json_to_gromet(gromet_path)
    print(gromet)

    extractions = AttributeCollection.from_json(attribute_collection)

    linker = GrometLinker(model_name=similarity_model, device=device, sim_threshold=similarity_threshold)

    linked_model = linker.link_model_to_text_extractions(gromet, extractions)

    if not output_path:
        input_gromet_name = str(Path(gromet_path).name)
        output_path = f'linked_{input_gromet_name}'

    with open(output_path, 'w') as f:
        json.dump(linked_model, f, default=str, indent=2)


def main():
    """ Module's entry point"""
    fire.Fire(link_gromet)


if __name__ == "__main__":
    main()
