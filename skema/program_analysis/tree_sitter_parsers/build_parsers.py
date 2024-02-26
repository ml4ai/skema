import argparse
import os
import yaml
import subprocess
import site
from pathlib import Path
from typing import List

from tree_sitter import Language, Parser

INSTALLED_LANGUAGES_FILEPATH = (
    Path(__file__).resolve().parent / "build" / "installed_languages.so"
)
LANGUAGES_YAML_FILEPATH = Path(__file__).parent / "languages.yaml"


def build_parsers(languages: List[str]) -> None:
    # The 'build' directory containing the cloned tree-sitter parsers and shard object library may or may not already exist.
    # We need to create it first if it does not exist
    language_build_dir = Path(INSTALLED_LANGUAGES_FILEPATH.parent)
    language_build_dir.mkdir(parents=True, exist_ok=True)

    language_yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    for language, language_dict in language_yaml_obj.items():
        language_clone_directory = language_build_dir / language_dict["tree_sitter_name"]
        if language in languages:
            # Clone the repository if it doesn't exist
            subprocess.run(
                ["git", "clone", language_dict["clone_url"]], cwd=language_build_dir
            )
            # Update the repository to pull any new commits
            subprocess.run(
                ["git", "pull"], cwd=language_clone_directory
            )
            # Checkout the correct commit if commit_sha is specified
            if language_dict.get("commit_sha"):
                subprocess.run(
                    ["git", "checkout", str(language_dict["commit_sha"])], cwd=language_clone_directory
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


def copy_to_site_packages():
    """Copy the .so file to the skema site package"""
    copy_path = Path(site.getsitepackages()[0]) / "skema" / "program_analysis" / "tree_sitter_parsers" / "build" / "installed_languages.so"  
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_bytes(INSTALLED_LANGUAGES_FILEPATH.read_bytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    language_yaml_obj = yaml.safe_load(open(LANGUAGES_YAML_FILEPATH))
    parser.add_argument("--all", action="store_true", help="Build all tree-sitter parsers")
    for language, language_dict in language_yaml_obj.items():
        flag = f"--{language}"
        help_text = f"Include {language} language"
        parser.add_argument(flag, action="store_true", help=help_text)
    parser.add_argument("--ci",  action="store_true", help="Copy to site packages if running on ci")
    args = parser.parse_args()

    if args.all:
        selected_languages = [language for language in yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())]
    else:
        selected_languages = [language for language, value in vars(args).items() if value]
    
    build_parsers(selected_languages)

    if args.ci:
        copy_to_site_packages()
