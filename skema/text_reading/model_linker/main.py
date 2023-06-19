# This is a sample Python script.
import json

import fire.fire_test
from askem_extractions.data_model import AttributeCollection

from linkers.petrinet import PetriNetLinker


def link_petrinet(petrinet: str, attribute_collection: str):
    with open(petrinet) as f:
        petrinet = json.load(f)

    extractions = AttributeCollection.from_json(attribute_collection)

    linker = PetriNetLinker(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None)

    linked_model = linker.link_model_to_text_extractions(petrinet, extractions)


    with open('petrinet_aligned.json', 'w') as f:
        json.dump(linked_model, f, default=str, indent=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fire.Fire(link_petrinet)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
