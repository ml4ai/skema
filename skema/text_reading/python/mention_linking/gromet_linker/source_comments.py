from pathlib import Path
from typing import Dict, List, Tuple
import json


class SourceComments:
    """
    line_comments are normal comments that you would find after a # in python.
    docstrings are comments situated between the function declaration and its body that are usually
        triple quoted.  Right now we don't have line numbers for them, so -1 is used.
    """

    def __init__(
        self,
        path: Path,
        line_comments: Dict[int, str],
        docstrings: Dict[str, List[str]],
    ):
        """
        This is a docstring for the __init__ method.
        """
        self.path = path
        self.line_comments = line_comments
        self.docstrings = docstrings
        self.line_docstrings = self.make_line_docstrings()

    def make_line_docstrings(self) -> Dict[str, List[Tuple[int, str]]]:

        # TODO: Until we get the real line numbers, they are all -1 and then filtered out later.
        return {
            key: [(-1, value) for value in values]
            for key, values in self.docstrings.items()
        }

    def file_name(self) -> str:
        return self.path.name

    def file_path(self) -> str:
        return str(self.path.absolute)

    def get_line_docstrings(self, name: str) -> List[Tuple[int, str]]:
        return self.line_docstrings.get(name, [])

    @staticmethod
    def from_file(comments_path: str) -> "SourceComments":
        """Reads the automatically extracted comments from the json file"""

        def _helper(data):
            line_comments = {
                line_comment[0]: line_comment[1]
                for line_comment in data["comments"]
            }
            docstrings = data["docstrings"]

            return SourceComments(
                path=Path(comments_path),
                line_comments=line_comments,
                docstrings=docstrings,
            )

        with open(comments_path) as file:
            data = json.load(file)

        # Make either a single instance of a dictionary or instances in the case of multiple files.
        if "comments" in data:
            return _helper(data)
        else:
            # TODO: What is this?
            return {key: _helper(value) for key, value in data.items()}
