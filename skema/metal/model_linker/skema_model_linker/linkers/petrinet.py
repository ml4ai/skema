from collections import defaultdict
from typing import Iterable, Dict, List, Any

from . import heuristics, AMRLinker
from ..walkers import JsonDictWalker, JsonNode
from ..walkers import PetriNetWalker


class PetriNetLinker(AMRLinker):

    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
        ret = defaultdict(list)
        for name, val, ix in elements:
            if (name == "states") or (name == "parameters" and 'name' in val):
                key = val['name'].strip()
                lower_case_key = key.lower()

                if "description" in val:
                    ret[f"{key}: {val['description']}"] = val
                else:
                    if lower_case_key in heuristics:
                        descs = heuristics[lower_case_key]
                        for desc in descs:
                            ret[f"{key}: {desc}"] = val
                ret[key] = val
            # elif name == "transitions":
            #     if "description" in val:
            #         ret[f"{val['id'].strip()}: {val['description']}"] = val
            #     else:
            #         ret[val['id'].strip()] = val

        return ret

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return PetriNetWalker(amr_data)
