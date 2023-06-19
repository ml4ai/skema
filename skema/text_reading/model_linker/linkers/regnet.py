from collections import defaultdict
from typing import Iterable, Dict, List, Any

from askem_extractions.data_model import Attribute, AnchoredExtraction

from linkers.amr_linker import AMRLinker
from walkers.json import JsonDictWalker, JsonNode
from walkers.petrinet import PetriNetWalker
from walkers.regnet import RegNetWalker


class RegNetLinker(AMRLinker):

    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
        ret = defaultdict(list)
        for name, val, ix in elements:
            id_ = val['id'].strip()
            if name == "vertices":
                rate_constant = val['rate_constant'].strip()
                if "rate_constant" in val:
                    if id_ != rate_constant:
                        ret[f"{id_}: {rate_constant}"] = val
                    else:
                        ret[id_] = val
                else:
                    ret[id_] = val
            elif name == "edges":
                if "properties" in val and "rate_constant" in val["properties"]:
                    rate_constant = val['properties']['rate_constant'].strip()
                    if id_ != rate_constant:
                        ret[f"{id_}: {rate_constant}"] = val
                    else:
                        ret[id_] = val
                else:
                    ret[id_] = val
        return ret

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return RegNetWalker(amr_data)


