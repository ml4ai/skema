import abc
from abc import ABC
from collections import defaultdict
from typing import Iterable, Dict, List, Any, Tuple, Optional, Union

import pandas as pd
import torch
from askem_extractions.data_model import Attribute, AnchoredEntity, AttributeCollection, AttributeType
from sentence_transformers import SentenceTransformer, util

from ..walkers import JsonNode, JsonDictWalker


class Linker(ABC):
    def __init__(self, model_name: str, sim_threshold: float = 0.7, device: Optional[str] = None):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._threshold = sim_threshold

        if device:
            self._model.to(device)

        self._model.eval()

    @abc.abstractmethod
    def _build_walker(self, amr_data: Dict[str, Any]) -> JsonDictWalker:
        pass

    @abc.abstractmethod
    def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
        """" Will generate candidate texts to link to text extractions """
        pass

    def _align_texts(self, sources: List[str], targets: List[str], threshold: float) -> List[Tuple[str, str]]:

        with torch.no_grad():
            s_embs = self._model.encode(sources)
            t_embs = self._model.encode(targets)

        similarities = util.pytorch_cos_sim(s_embs, t_embs)

        indices = (similarities >= threshold).nonzero()

        ret = list()
        for ix in indices:
            ret.append((sources[ix[0]], targets[ix[1]]))

        return ret

    def _generate_linking_targets(self, extractions: Iterable[Attribute]) -> Dict[str, List[AnchoredEntity]]:
        """ Will generate candidate texts to link to model elements """
        ret = defaultdict(list)
        for ex in extractions:
            for name in ex.payload.mentions:
                if len(ex.payload.text_descriptions) > 0:
                    for desc in ex.payload.text_descriptions:
                        ret[f"{name.name.strip()}: {desc.description.strip()}"].append(ex)
                        ret[desc.description.strip()].append(ex)
                else:
                    candidate_text = f"{name.name.strip()}"
                    ret[candidate_text].append(ex)
        return ret

    @abc.abstractmethod
    def link_model_to_text_extractions(self, data: Union[Any, Dict[str, Any]], extractions: AttributeCollection) -> \
    Dict[str, Any]:
        pass


class AMRLinker(Linker, ABC):

    def link_model_to_manual_annotations(self, data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
            Similarly to linking a model to text extractions. This will link it to ground truth extractions
            Used mostly for debugging
        """

        # Make a copy of the amr to avoid mutating the original model
        data = {**data}

        # Filter out the targets from the annotations
        targets = defaultdict(list)
        for candidate in candidates:
            if candidate['type'] == "Highlight" and candidate['color'] == "#f9cd59": # This color is an anchored extraction
                key = candidate["text"]
                targets[key].append(candidate)


        walker = self._build_walker(data)

        to_link = list(walker.walk())
        sources = self._generate_linking_sources(to_link)

        pairs = self._align_texts(list(sources.keys()), list(targets.keys()), threshold=self._threshold)

        linked_targets = list()
        for s_key, t_key in pairs:
            source = sources[s_key]
            target = targets[t_key]

            # Get the AMR ID of the source and add it to the target extractions
            for t in target:
                t['amr_element_id'] = source['id']
                linked_targets.append(t)

        # Serialize the attribute collection to json, after alignment
        data["metadata"] = linked_targets

        return data

    def link_model_to_text_extractions(self, data: Dict[str, Any], extractions: AttributeCollection) -> Dict[str, Any]:

        # Make a copy of the amr to avoid mutating the original model
        data = {**data}

        targets = self._generate_linking_targets(
            e for e in extractions.attributes if e.type == AttributeType.anchored_entity)

        walker = self._build_walker(data)

        to_link = list(walker.walk())
        sources = self._generate_linking_sources(to_link)

        pairs = self._align_texts(list(sources.keys()), list(targets.keys()), threshold=self._threshold)

        for s_key, t_key in pairs:
            source = sources[s_key]
            target = targets[t_key]

            # Get the AMR ID of the source and add it to the target extractions
            for t in target:
                t.amr_element_id = source['id']

        # Serialize the attribute collection to json, after alignment
        attribute_dict = extractions.model_dump(exclude_unset=True)
        data["metadata"] = attribute_dict

        return data
