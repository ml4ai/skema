import re
from typing import List, Tuple

class LanguageMatcher:
    def match_line_comment(
        self, name: str, line_comments: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        return list()


class FortranMatcher(LanguageMatcher):
    pass


class PythonMatcher(LanguageMatcher):
    def __init__(self):
        self.separator = r"[:,\.\s]"

    def match_line_comment(
        self, name: str, line_comments: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        # TODO: Use re.escape() after regression testing
        # pattern = self.separator + re.escape(name) + self.separator
        pattern = self.separator + name + self.separator
        matches = [
            (line, comment)
            for (line, comment) in line_comments
            if re.search(pattern, comment)
        ]
        return matches


class VariableNameMatcher(LanguageMatcher):
    def __init__(self, language: str):
        if language == "python":
            self.matcher = PythonMatcher()
        elif language == "fortran":
            self.matcher = FortranMatcher()
        else:
            self.matcher = PythonMatcher()  # TODO: raise an exception?

    def match_line_comment(
        self, name: str, line_comments: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        if name:
            return self.matcher.match_line_comment(name, line_comments)
        else:
            return []
