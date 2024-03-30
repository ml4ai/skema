from collections import defaultdict
from typing import Iterable, Dict, List, Any

from . import heuristics, AMRLinker
from ..walkers import JsonDictWalker, JsonNode
from ..walkers import GeneralizedAMRWalker


class GeneralizedAMRLinker(AMRLinker):

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
                        descriptions = heuristics[lower_case_key]
                        for desc in descriptions:
                            ret[f"{key}: {desc}"] = val
                ret[key] = val

        return ret

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return GeneralizedAMRWalker(amr_data)
