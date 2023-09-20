from collections import defaultdict
from typing import Iterable, Dict, List, Any

from . import heuristics, AMRLinker
from ..walkers import JsonDictWalker, JsonNode
from ..walkers import PetriNetWalker


class PetriNetLinker(AMRLinker):

    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
        ret = defaultdict(list)
        for name, val, ix in elements:
            if name == "states":
                key = val['name'].strip().lower()

                if "description" in val:
                    ret[f"{val['name'].strip()}: {val['description']}"] = val
                else:
                    if key in heuristics:
                        descs = heuristics[key]
                        for desc in descs:
                            ret[f"{val['name'].strip()}: {desc}"] = val
                    ret[val['name'].strip()] = val
            elif name == "parameters" and 'id' in val:
                ret[val['id'].strip()] = val
            # elif name == "transitions":
            #     if "description" in val:
            #         ret[f"{val['id'].strip()}: {val['description']}"] = val
            #     else:
            #         ret[val['id'].strip()] = val

        return ret

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return PetriNetWalker(amr_data)
