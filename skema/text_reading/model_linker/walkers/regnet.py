from typing import Optional, Any

from walkers.json import JsonDictWalker


class RegNetWalker(JsonDictWalker):

    def _filter(self, obj_name: Optional[str], obj: Any, index: Optional[int]) -> bool:
        return obj_name in {"vertices", "edges", "items"}
