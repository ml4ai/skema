from typing import Any, Optional

from skema.gromet.fn import GrometFNModuleCollection, GrometFNModule, GrometFN
from . import ModelWalker

import itertools as it


class GrometWalker(ModelWalker):
    def __init__(self, gfn: GrometFNModuleCollection):
        self.__gfn = gfn # This is the gromet

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

    def _walk_fn(self, fn:GrometFN, callback=None, **kwargs):
        fn.
    def _walk_module(self, module_name: Optional[str], module: GrometFNModule, callback=None, **kwargs):

        return []

    def walk(self, callback=None, *args, **kwargs):
        """ Start the walk from the root of the json object """
        return list(it.chain.from_iterable(self._walk_module(module) for module in self.__gfn.modules))
        # return reversed(self.__step(obj_name=None, obj=self.__gfn, callback=callback, **kwargs))