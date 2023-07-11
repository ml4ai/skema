from collections import defaultdict
from typing import Dict, Any, Union, Iterable, List

from askem_extractions.data_model import AttributeCollection

from .amr_linker import AMRLinker, Linker
from ..walkers import JsonDictWalker, JsonNode


class GrometLinker(Linker):

	def link_model_to_text_extractions(self, data: Union[Any, Dict[str, Any]], extractions: AttributeCollection) -> \
	Dict[str, Any]:
		""" Here the assignment takes place """
		pass

	def _generate_linking_sources(self, elements: Iterable[JsonNode]) -> Dict[str, List[Any]]:
		ret = defaultdict(list)
		""" Use the gromet structure to get the comments and variable names for grounding """
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

	def _build_walker(self, gromet) -> JsonDictWalker:
		""" Walk and select the linkable elements of the Gromet """
		return GrometWalker(gromet)
