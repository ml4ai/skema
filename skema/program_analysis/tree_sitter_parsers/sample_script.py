from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH
from tree_sitter import Language

language = Language(INSTALLED_LANGUAGES_FILEPATH, "fortran")
print(language)