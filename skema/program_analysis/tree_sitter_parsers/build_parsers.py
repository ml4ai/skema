import argparse
import os
import yaml
import subprocess
from pathlib import Path
from typing import List

from tree_sitter import Language, Parser

INSTALLED_LANGUAGES_FILEPATH = (
    Path(__file__).parent / "build" / "installed_languages.so"
)
LANGUAGES_YAML_FILEPATH = Path(__file__).parent / "languages.yaml"


def build_parsers(languages: List[str]) -> None:
    # The 'build' directory containing the cloned tree-sitter parsers and shard object library may or may not already exist.
    # We need to create it first if it does not exist
    language_build_dir = Path(INSTALLED_LANGUAGES_FILEPATH.parent)
    language_build_dir.mkdir(parents=True, exist_ok=True)

    language_yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    for language, language_dict in language_yaml_obj.items():
        if language in languages:
            subprocess.run(
                ["git", "clone", language_dict["clone_url"]], cwd=language_build_dir
            )

    # We can set pass the cwd for subprocess as an argument to run().
    # However, Language.build_library requires the cwd to be set to the build directory
    wd = os.getcwd()
    os.chdir(language_build_dir)
    # If the library file already exists, build_library will fail and return False.
    # So, we must check if it exists and delete it first
    INSTALLED_LANGUAGES_FILEPATH.unlink(missing_ok=True)
    Language.build_library(
        # Store the library in the `build` directory
        str(INSTALLED_LANGUAGES_FILEPATH.name),
        # Include one or more languages
        [
            language_dict["tree_sitter_name"]
            for language, language_dict in language_yaml_obj.items()
            if language in languages
        ],
    )

    os.chdir(wd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    language_yaml_obj = yaml.safe_load(open(LANGUAGES_YAML_FILEPATH))
    for language, language_dict in language_yaml_obj.items():
        flag = f"--{language}"
        help_text = f"Include {language} language"
        parser.add_argument(flag, action="store_true", help=help_text)

    args = parser.parse_args()
    selected_languages = [language for language, value in vars(args).items() if value]
    build_parsers(selected_languages)
