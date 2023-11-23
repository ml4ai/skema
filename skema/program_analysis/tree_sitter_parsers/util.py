import yaml
from typing import Dict

from skema.program_analysis.tree_sitter_parsers.build_parsers import LANGUAGES_YAML_FILEPATH

def generate_extension_to_language_dict() -> Dict:
    """Create a dictionary mapping between supported file extensions and language name"""
    yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    return {extension: language for language, language_dict in yaml_obj.items() for extension in language_dict["extensions"]}

EXTENSION_DICT = generate_extension_to_language_dict()

def extension_to_language(extension: str) -> str:
    """Given an extension, return the language name or None if its an unsupported extension"""
    return EXTENSION_DICT.get(extension, None)

