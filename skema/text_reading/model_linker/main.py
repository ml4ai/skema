# This is a sample Python script.
import json
from pathlib import Path
from typing import Optional

import fire.fire_test
from askem_extractions.data_model import AttributeCollection

from linkers.petrinet import PetriNetLinker


def link_amr(
        amr_path: str,                                                     # Path of the AMR model
        attribute_collection: str,                                         # Path to the attribute collection
        amr_type: str,                                                     # AMR model type. I.e. "petrinet" or "regnet"
        output_path: Optional[str] = None,                                        # Output file path
        similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # Transformer model to compute similarities
        device: Optional[str] = None                                       # PyTorch device to run the model on
        ):
    """ Links and AMR model to an attribute collections from ASKEM text reading pipelines """

    if amr_type == "petrinet":
        Linker = PetriNetLinker
    # elif amr_type == "regnet":
    #     Linker = RegNetLinker
    else:
        raise NotImplementedError(f"{amr_type} AMR currently not supported")

    with open(amr_path) as f:
        amr = json.load(f)

    extractions = AttributeCollection.from_json(attribute_collection)

    linker = Linker(model_name=similarity_model, device=device)

    linked_model = linker.link_model_to_text_extractions(amr, extractions)

    if not output_path:
        input_amr_name = str(Path(amr_path).name)
        output_path = f'linked_{input_amr_name}'

    with open(output_path, 'w') as f:
        json.dump(linked_model, f, default=str, indent=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fire.Fire(link_amr)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
