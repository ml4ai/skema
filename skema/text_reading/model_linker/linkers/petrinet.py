from collections import defaultdict
from typing import Iterable, Dict, List, Any

from askem_extractions.data_model import Attribute, AnchoredExtraction

from linkers.amr_linker import AMRLinker
from walkers.json import JsonDictWalker, JsonNode
from walkers.petrinet import PetriNetWalker


class PetriNetLinker(AMRLinker):

    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
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

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return PetriNetWalker(amr_data)


