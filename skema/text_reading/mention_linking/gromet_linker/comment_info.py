from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union


class CommentInfo(NamedTuple):
    line_range: Optional[range]
    name: Optional[str]
    line_docstrings: List[tuple[int, str]]
    comments: List[Union[str, Tuple[int, str]]]
    mentions: Mapping[str, Any]

    def print(self) -> None:
        docstrings = [docstring for _, docstring in self.line_docstrings]
        print("===================")
        print(f"{self.line_range} {self.name if self.name else ''}:")
        print()
        print("\n".join(docstrings))
        print(
            "\n".join(c[1] if type(c) == tuple else c for c in self.comments)
        )
        if len(self.mentions) > 0:
            print(
                "Aligned mentions:\n"
                + "\n".join(f"{s}: {m['text']}" for m, s in self.mentions)
            )
        print()
