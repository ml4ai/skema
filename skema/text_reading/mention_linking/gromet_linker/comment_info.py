from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union

class CommentInfo(NamedTuple):
	line_range: Optional[range]
	name: Optional[str]
	docstring: List[str]
	comments: List[Union[str, Tuple[int, str]]]
	mentions: Mapping[str, Any]

	def print(self) -> None:
		print("===================")
		print(f"{self.line_range} {self.name if self.name else ''}:")
		print()
		print("\n".join(self.docstring))
		print("\n".join(c[1] if type(c) == tuple else c for c in self.comments))
		if len(self.mentions) > 0:
			print("Aligned mentions:\n" + '\n'.join(f"{s}: {m['text']}" for m, s in self.mentions))
		print()
