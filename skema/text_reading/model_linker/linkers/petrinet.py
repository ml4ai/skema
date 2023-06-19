from collections import defaultdict
from typing import Iterable, Dict, List

from askem_extractions.data_model import Attribute, AnchoredExtraction

from linkers.amr_linker import AMRLinker
from walkers.json import JsonDictWalker
from walkers.petrinet import PetriNetWalker


class PetriNetLinker(AMRLinker):

    def _build_walker(self, amr_data) -> JsonDictWalker:
        return PetriNetWalker(amr_data)

    def _generate_linking_targets(self, extractions: Iterable[Attribute]) -> Dict[str, List[AnchoredExtraction]]:
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
