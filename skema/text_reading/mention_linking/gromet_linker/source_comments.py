from dataclasses import dataclass
from pathlib import Path

import json

@dataclass
class SourceComments:
	path: Path
	line_comments: dict[int, str]
	doc_strings: dict[str, list[str]]

	@property
	def file_name(self) -> str:
		return self.path.name

	@property
	def file_path(self) -> str:
		return str(self.path.absolute)

	@staticmethod
	def from_file(comments_path: str) -> "SourceComments":
		""" Reads the automatically extracted comments from the json file """

		def _helper(data):
			line_comments = {line_comment[0]: line_comment[1] for line_comment in data['comments']}
			doc_strings = data['docstrings']

			return SourceComments(
				path = Path(comments_path),
				line_comments = line_comments,
				doc_strings = doc_strings
			)

		with open(comments_path) as file:
			data = json.load(file)

		# Make either a single instance of a dictionary or instances in the case of multiple files
		if "comments" in data:
			return _helper(data)
		else:
			return {key: _helper(value) for key, value in data.items()}

