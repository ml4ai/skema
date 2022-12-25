import re

class LanguageMatcher():

	def match(self, name: str, comments):
		return list()

class PythonMatcher(LanguageMatcher):

	def match(self, name: str, comments):
		result = list()
		try:
			pattern = re.compile(r"[:,\.\s]" + name + r"[:,\.\s]")
		except Exception as	e: # TODO Escape the name.
			# print(e)
			pass
		else:
			for l in comments:
				if len(pattern.findall(l)) > 0:
					result.append(l)

		return result

class FortranMatcher(LanguageMatcher):
	pass

class VariableNameMatcher(LanguageMatcher):

	def __init__(self, language: str):
		if language == "python":
			self.matcher = PythonMatcher()
		elif language == "fortran":
			self.matcher = FortranMatcher()
		else:
			self.matcher = PythonMatcher() # TODO: raise an exception?

	def match(self, name: str, comments):
		return self.matcher.match(name, comments)
