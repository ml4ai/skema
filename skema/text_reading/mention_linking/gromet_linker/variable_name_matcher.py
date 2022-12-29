import re

class LanguageMatcher():

	def match(self, name: str, comments: list[str]) -> list[str]:
		return list()

class FortranMatcher(LanguageMatcher):
	pass

class PythonMatcher(LanguageMatcher):
	def __init__(self):
		self.separator = r"[:,\.\s]"

	def match(self, name: str, comments: list[str]) -> list[str]:
		# TODO: Use re.escape() after regression testing
		# pattern = self.separator + re.escape(name) + self.separator
		pattern = self.separator + name + self.separator
		matches = [comment for comment in comments if re.search(pattern, comment)]
		return matches

class VariableNameMatcher(LanguageMatcher):

	def __init__(self, language: str):
		if language == "python":
			self.matcher = PythonMatcher()
		elif language == "fortran":
			self.matcher = FortranMatcher()
		else:
			self.matcher = PythonMatcher() # TODO: raise an exception?

	def match(self, name: str, comments: list[str]) -> list[str]:
		if name:
			return self.matcher.match(name, comments)
		else:
			return list()
