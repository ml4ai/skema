import argparse
import re
import os
import shutil
import logging
from typing import List, Optional
from pathlib import Path
from subprocess import run, PIPE

from tree_sitter import Parser, Node, Language, Tree

from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH,
)
from skema.program_analysis.CAST.fortran.preprocessor.fixed2free import convertToFree


def preprocess(
    source_path: Path,
    out_dir=None,
    overwrite=False,
    out_missing_includes=False,
    out_gcc=False,
    out_unsupported=False,
    out_free=False,
) -> str:
    """Run the full preprocessing pipeline for Fortran->Tree-Sitter->CAST
    Takes the original source as input and will return the tree-sitter parse tree as output
    An intermediary directory will also be created containing:
    1. A log of missing files that are included by preprocessor directives in the source code
    2. The intermediary product from running the c-preprocessor
    3. A log of unsupported idioms
    4. The source code converted to free-form
    """
    # NOTE: The order of preprocessing steps does matter. We have to run the GCC preprocessor before correcting the continuation lines or there could be issues

    source = source_path.read_text()

    # Get paths for intermediate products
    if out_dir:
        if not (out_missing_includes or out_gcc or out_unsupported or out_free):
            logging.warning("out_dir is specified, but no out flags are set")

        out_dir.mkdir(parents=True, exist_ok=True)

        missing_includes_path = Path(out_dir, "missing_includes.txt")
        gcc_path = Path(out_dir, "gcc.F")
        unsupported_path = Path(out_dir, "unsupported_idioms.txt")
        free_path = Path(out_dir, "corrected.F")
        parse_path = Path(out_dir, "parse_tree.txt")

    # Step 1: Check for missing included files
    # Many source files won't have includes. We only need to check missing includes if a source contains an include statement.
    if len(produce_include_summary(source)) > 0:
        missing_includes = check_for_missing_includes(source_path)
        if out_missing_includes:
            missing_includes_path.write_text("\n".join(missing_includes))

        if len(missing_includes) > 0:
            logging.error("Missing required included files, missing files were:")
            for include in missing_includes:
                logging.error(include)
            exit()
    elif out_missing_includes:
        missing_includes_path.write_text("Source file contains no include statements")

    # Step 2: Correct include directives to remove system references
    source = fix_include_directives(source)

    # Step 3: Process with gcc c-preprocessor
    source = run_c_preprocessor(source, source_path.parent)
    if out_gcc:
        gcc_path.write_text(source)

    # Step 4: Prepare for tree-sitter
    # This step removes any additional preprocessor directives added or not removed by GCC
    source = "\n".join(
        ["!" + line if line.startswith("#") else line for line in source.splitlines()]
    )

    # Step 5: Check for unsupported idioms
    if out_unsupported:
        unsupported_path.write_text(
            "\n".join(search_for_unsupported_idioms(source, "idioms_regex.txt"))
        )

    # Step 6 : Convert to free-form for tree-sitter parsing
    source = convert_to_free_form(source)
    if out_free:
        free_path.write_text(source)

    return source


def produce_include_summary(source: str) -> List:
    """Uses regex to produce a list of all included files in a source"""
    includes = []

    system_re = "#include\s+<(.*)>"
    local_re = '#include\s+"(.*)"'

    for match in re.finditer(system_re, source):
        includes.append(match.group(1))
    for match in re.finditer(local_re, source):
        includes.append(match.group(1))

    return includes


def check_for_missing_includes(source_path: Path):
    """Gathers all required includes and check if they have been added to the include_SOURCE directory"""

    missing_files = []

    # First we will check for the include directory
    include_base_directory = Path(source_path.parent, f"include_{source_path.stem}")
    if not include_base_directory.exists():
        missing_files.append(include_base_directory)
        return missing_files

    # Add original source to includes directory
    shutil.copy2(source_path, include_base_directory)

    # Next gather all includes in each source file
    includes = []
    for dirpath, dirnames, filenames in os.walk(include_base_directory):
        for file in filenames:
            file_source = Path(dirpath, file).read_text()
            includes.extend(produce_include_summary(file_source))

    # Check for missing files
    already_checked = set()
    for include in includes:
        if include in already_checked:
            continue
        if not Path(include_base_directory, include).exists():
            missing_files.append(include)
        already_checked.add(include)
    return missing_files


def search_for_unsupported_idioms(source: str, idioms_regex_path: str):
    """Check source string for unsupported idioms using regex. Returns a log of found matches as well as line information"""
    log = []
    lines = open(idioms_regex_path, "r").read().splitlines()
    for line in lines:
        for match in re.finditer(line, source, flags=re.MULTILINE):
            line_number = source[: match.span()[0]].count("\n")
            log.append(f"Found unsupported idiom matching regex: {line}")
            log.append(f"Match was: {match.group(0)}")
            log.append(f"Line was: {line_number}")
    return log


def fix_include_directives(source: str) -> str:
    """There are a few corrections we need to make to the include statements
    1. Convert system level includes to local includes
    """
    processed_lines = []
    for i, line in enumerate(source.splitlines()):
        if "#include" in line:
            line = line.replace("<", '"').replace(">", '"')
        processed_lines.append(line)
    source = "\n".join(processed_lines)

    return source


def run_c_preprocessor(source: str, include_base_path: Path) -> str:
    """Run the gcc c-preprocessor. Its run from the context of the include_base_path, so that it can find all included files"""
    result = run(
        ["gcc", "-cpp", "-E", "-"],
        input=source,
        text=True,
        capture_output=True,
        universal_newlines=True,
        cwd=include_base_path,
    )
    return result.stdout


def convert_to_free_form(source: str) -> str:
    """If fixed-form Fortran source, convert to free-form"""

    def validate_parse_tree(source: str) -> bool:
        """Parse source with tree-sitter and check if an error is returned."""
        language = Language(INSTALLED_LANGUAGES_FILEPATH, "fortran")
        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(source, encoding="utf-8"))
        return "ERROR" not in tree.root_node.sexp()

    # We don't know for sure if a source is meant to be fixed-form or free-form
    # So, we will run the parser first to check
    if validate_parse_tree(source):
        return source
    else:
        # convertToFree takes a stream as input and returns a generator
        free_source = "".join(
            [line for line in convertToFree(source.splitlines(keepends=True))]
        )
        if validate_parse_tree(free_source):
            return free_source

    return source


def main():
    """Run the preprocessor as a script"""
    parser = argparse.ArgumentParser(description="Fortran preprocessing script")
    parser.add_argument("source_path", type=str, help="Path to the source file")
    parser.add_argument("out_dir", type=str, help="Output directory path")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--out_missing_includes",
        action="store_true",
        help="Output missing includes log",
    )
    parser.add_argument(
        "--out_gcc",
        action="store_true",
        help="Output source after running the GCC preprocessor",
    )
    parser.add_argument(
        "--out_unsupported",
        action="store_true",
        help="Output unsupported idioms log",
    )
    parser.add_argument(
        "--out_free",
        action="store_true",
        help="Output source after fixing unsupported idioms",
    )
    args = parser.parse_args()

    preprocess(
        Path(args.source_path),
        Path(args.out_dir),
        args.overwrite,
        args.out_missing_includes,
        args.out_gcc,
        args.out_unsupported,
        args.out_free,
    )


if __name__ == "__main__":
    main()
