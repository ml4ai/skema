# This is a sample Python script.
import abc
import json
from abc import ABC
from collections import defaultdict
from typing import Dict, Any, Optional, List, Iterable, NamedTuple, Tuple

import fire.fire_test
from askem_extractions.data_model import AttributeCollection, AnchoredExtraction, AttributeType, Attribute
from sentence_transformers import SentenceTransformer, util


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class ModelWalker(ABC):
    """ Defines the interface to walk through a model to attach text extractions """

    def __iter__(self):
        return iter(self)

    @abc.abstractmethod
    def walk(self, callback=None, *args, **kwargs):
        pass


class JsonNode(NamedTuple):
    name: str
    val: Any
    index: int


class JsonDictWalker(ModelWalker):

    def __init__(self, data: Dict[str, Any]):
        self.__data = data

    def __iter__(self):
        return self.walk()

    def _filter(self, obj_name: Optional[str], obj: Any, index: Optional[int]) -> bool:
        """ Decides whether traverse the current element.
            True by default. Override to customize behavior """
        return True

    def __step(self, obj_name: Optional[str], obj: Any, index: Optional[int] = None, callback=None, **kwargs):
        """ Walks over the elements of the json dictionary """

        ret = list()

        allowed = self._filter(obj_name, obj, index)

        # If a callback is provided, call it
        if allowed and callback:
            callback(obj_name, obj, index)

        for prop, val in obj.items():
            if isinstance(val, list):
                for ix, elem in enumerate(val):
                    if type(elem) in (list, dict):
                        ret.extend(self.__step(prop, elem, ix, callback, **kwargs))
            elif isinstance(val, dict):
                ret.extend(self.__step(prop, val, None, callback, **kwargs))

        if allowed:
            ret.append(JsonNode(obj_name, obj, index))

        return ret

    def walk(self, callback=None, *args, **kwargs):
        """ Start the walk from the root of the json object """
        return reversed(self.__step(obj_name=None, obj=self.__data, callback=callback, **kwargs))


class PetriNetWalker(JsonDictWalker):

    def _filter(self, obj_name: Optional[str], obj: Any, index: Optional[int]) -> bool:
        return obj_name in {"states", "transitions"}


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.to("mps")
model.eval()


def generate_linking_targets(extractions: Iterable[Attribute]) -> Dict[str, List[AnchoredExtraction]]:
    """ Will generate candidate texts to link to model elements """
    ret = defaultdict(list)
    for ex in extractions:
        for name in ex.payload.names:
            if len(ex.payload.descriptions) > 0:
                for desc in ex.payload.descriptions:
                    ret[f"{name.name.strip()}: {desc.source.strip()}"].append(ex)
                    ret[desc.source.strip()].append(ex)
            else:
                candidate_text = f"{name.name.strip()}"
                ret[candidate_text].append(ex)
    return ret


def generate_linking_sources(elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
    """" Will generate candidate texts to link to text extractions """
    ret = defaultdict(list)
    for name, val, ix in elements:
        if name == "states":
            if "description" in val:
                ret[f"{val['name'].strip()}: {val['description']}"] = val
            else:
                ret[val['name'].strip()] = val
        elif name == "transitions":
            if "description" in val:
                ret[f"{val['id'].strip()}: {val['description']}"] = val
            else:
                ret[val['id'].strip()] = val
    return ret


def align_texts(sources: List[str], targets: List[str], threshold: float = 0.7) -> List[Tuple[str, str]]:
    s_embs = model.encode(sources)
    t_embs = model.encode(targets)

    similarities = util.pytorch_cos_sim(s_embs, t_embs)

    indices = (similarities >= threshold).nonzero()

    ret = list()
    for ix in indices:
        ret.append((sources[ix[0]], targets[ix[1]]))

    return ret


def link_petrinet(petrinet: str, attribute_collection: str):
    with open(petrinet) as f:
        petrinet = json.load(f)

    attribute_collection = AttributeCollection.from_json(attribute_collection)
    targets = generate_linking_targets(
        e for e in attribute_collection.attributes if e.type == AttributeType.anchored_extraction)

    walker = PetriNetWalker(petrinet)

    to_link = list(walker.walk())
    sources = generate_linking_sources(to_link)

    pairs = align_texts(list(sources.keys()), list(targets.keys()))

    for s_key, t_key in pairs:
        source = sources[s_key]
        target = targets[t_key]

        # Get the AMR ID of the source and add it to the target extractions
        for t in target:
            t.amr_element_id = source['id']

    # Serialize the attribute collection to json, after alignment
    attribute_dict = attribute_collection.dict(exclude_unset=True)
    petrinet["metadata"] = attribute_dict
    with open('output.json', 'w') as f:
        json.dump(petrinet, f, default=str, indent=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fire.Fire(link_petrinet)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
