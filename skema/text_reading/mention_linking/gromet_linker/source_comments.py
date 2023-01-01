from pathlib import Path

import json

class SourceComments():
	
	def __init__(self, path: Path, line_comments: dict[int, str], doc_strings: dict[str, list[str]]):
		self.path = path
		self.line_comments = line_comments
		self.doc_strings = doc_strings
		self.line_docstrings = self.make_line_docstrings()

	def make_line_docstrings(self) -> dict[str, list[tuple[int, str]]]:

		# TODO: Until we get the real line numbers, they are all -1 and then filtered out later.
		return {key: [(-1, value) for value in values] for key, values in self.doc_strings.items()}

	def file_name(self) -> str:
		return self.path.name

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

		# Make either a single instance of a dictionary or instances in the case of multiple files.
		if "comments" in data:
			return _helper(data)
		else:
			# TODO: What is this?
			return {key: _helper(value) for key, value in data.items()}
