from dataclasses import dataclass
from pathlib import Path

import json

@dataclass
class SourceComments:
	path:Path
	line_comments:dict[int, str]
	doc_strings:dict[str, str]

	@property
	def file_name(self) -> str:
		return self.path.name

	@property
	def file_path(self) -> str:
		return str(self.path.absolute)

	@staticmethod
	def from_file(comments_path:str) -> "SourceComments":
		""" Reads the automatically extracted comments from the json file """

		def _helper(data):
			line_comments = {l[0]:l[1] for l in data['comments']}
			doc_strings = data['docstrings']

			return SourceComments(
				path = Path(comments_path),
				line_comments= line_comments,
				doc_strings= doc_strings
			)

		with open(comments_path) as f:
			data = json.load(f)

		# Make either a single instance of a dictionary or instances in the case of multiple files
		if "comments" in data:
			return _helper(data)
		else:
			return {k:_helper(v) for k, v in data.items()}
			
