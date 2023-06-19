import abc
from abc import ABC
from collections import defaultdict
from typing import Iterable, Dict, List, Any, Tuple, Optional

import torch
from askem_extractions.data_model import Attribute, AnchoredExtraction, AttributeCollection, AttributeType
from sentence_transformers import SentenceTransformer, util

from walkers.json import JsonNode, JsonDictWalker


class AMRLinker(ABC):

    def __init__(self, model_name: str, sim_threshold: float = 0.7, device: Optional[str] = None):
        self.__model_name = model_name
        self.__model = SentenceTransformer(model_name)
        self.__threshold = sim_threshold

        if device:
            self.__model.to(device)

        self.__model.eval()

    @abc.abstractmethod
    def _build_walker(self, amr_data:Dict[str, Any]) -> JsonDictWalker:
        pass


    def __generate_linking_targets(self, extractions: Iterable[Attribute]) -> Dict[str, List[AnchoredExtraction]]:
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

    @abc.abstractmethod
    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
        """" Will generate candidate texts to link to text extractions """
        pass

    def __align_texts(self, sources: List[str], targets: List[str], threshold: float) -> List[Tuple[str, str]]:

        with torch.no_grad():
            s_embs = self.__model.encode(sources)
            t_embs = self.__model.encode(targets)

        similarities = util.pytorch_cos_sim(s_embs, t_embs)

        indices = (similarities >= threshold).nonzero()

        ret = list()
        for ix in indices:
            ret.append((sources[ix[0]], targets[ix[1]]))

        return ret

    def link_model_to_text_extractions(self, amr_data: Dict[str, Any], extractions: AttributeCollection) -> Dict[str, Any]:

        # Make a copy of the amr to avoid mutating the original model
        amr_data = {**amr_data}

        targets = self.__generate_linking_targets(
            e for e in extractions.attributes if e.type == AttributeType.anchored_extraction)

        walker = self._build_walker(amr_data)

        to_link = list(walker.walk())
        sources = self._generate_linking_sources(to_link)

        pairs = self.__align_texts(list(sources.keys()), list(targets.keys()), threshold=self.__threshold)

        for s_key, t_key in pairs:
            source = sources[s_key]
            target = targets[t_key]

            # Get the AMR ID of the source and add it to the target extractions
            for t in target:
                t.amr_element_id = source['id']

        # Serialize the attribute collection to json, after alignment
        attribute_dict = extractions.dict(exclude_unset=True)
        amr_data["metadata"] = attribute_dict

        return amr_data


