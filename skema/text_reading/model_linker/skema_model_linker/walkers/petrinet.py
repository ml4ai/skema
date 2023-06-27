from typing import Optional, Any

from . import JsonDictWalker


class PetriNetWalker(JsonDictWalker):

    def _filter(self, obj_name: Optional[str], obj: Any, index: Optional[int]) -> bool:
        return obj_name in {"states", "transitions"}
