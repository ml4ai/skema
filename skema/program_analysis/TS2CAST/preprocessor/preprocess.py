import re
import os
import shutil
from typing import List
from subprocess import run, PIPE

from tree_sitter import Parser, Node, Language, Tree

from skema.program_analysis.TS2CAST.build_tree_sitter_fortran import (
    LANGUAGE_LIBRARY_REL_PATH,
)


def preprocess(
    source_path: str,
    out_path: str,
    overwrite=False,
    out_missing_includes=False,
    out_gcc=False,
    out_unsupported=False,
    out_corrected=False,
    out_parse=False,
) -> Tree:
    """Run the full preprocessing pipeline for Fortran->Tree-Sitter->CAST
    Takes the original source as input and will return the tree-sitter parse tree as output
    An intermediary directory will also be created containing:
    1. A log of missing files that are included by preprocessor directives in the source code
    2. The intermediary product from running the c-preprocessor
    3. A log of unsupported idioms
    4. The source code with unsupported idioms corrected
    5. The tree-sitter parse tree
    """

    source = open(source_path, "r").read()
    source_file_name = os.path.basename(source_path).split(".")[0]
    source_directory = os.path.dirname(source_path)
    include_base_directory = os.path.join(
        source_directory, f"include_{source_file_name}"
    )

    # NOTE: The order of preprocessing steps does matter. We have to run the GCC preprocessor before correcting the continuation lines or there could be issues
    try:
        os.mkdir(out_path)
    except FileExistsError:
        if not overwrite:
            exit()

    # Step 1: Check for missing included files
    missing_includes = check_for_missing_includes(source_path)
    if out_missing_includes:
        missing_includes_path = os.path.join(out_path, "missing_includes.txt")
        with open(missing_includes_path, "w") as f:
            f.write("\n".join(missing_includes))
    if len(missing_includes) > 0:
        print("Missing required included files, missing files were:")
        for include in missing_includes:
            print(include)
        exit()

    # Step 2: Correct include directives to remove system references
    source = fix_include_directives(source)

    # Step 3: Process with gcc c-preprocessor
    source = run_c_preprocessor(source, include_base_directory)
    if out_gcc:
        gcc_path = os.path.join(out_path, "gcc.F")
        with open(gcc_path, "w") as f:
            f.write(source)

    # Step 4: Prepare for tree-sitter
    # This step removes any additional preprocessor directives added or not removed by GCC
    source = "\n".join(
        ["!" + line if line.startswith("#") else line for line in source.splitlines()]
    )

    # Step 5: Check for unsupported idioms
    if out_unsupported:
        unsupported_path = os.path.join(out_path, "unsupported_idioms.txt")
        with open(unsupported_path, "w") as f:
            f.writelines(search_for_unsupported_idioms(source, "idioms_regex.txt"))

    # Step 6 : Fix unsupported idioms
    source = fix_unsupported_idioms(source)
    if out_corrected:
        corrected_path = os.path.join(out_path, "corrected.F")
        with open(corrected_path, "w") as f:
            f.write(source)

    # Stage 7: Parse with tree-sitter
    parse_tree = tree_sitter_parse(source)
    if out_parse:
        parse_path = os.path.join(out_path, "parse_tree.txt")
        with open(parse_path, "w") as f:
            f.write(parse_tree.root_node.sexp())

    return parse_tree


def check_for_missing_includes(source_path: str):
    """Gathers all required includes and check if they have been added to the include_SOURCE directory"""

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

    missing_files = []

    # First we will check for the include directory
    source_file_name = os.path.basename(source_path).split(".")[0]
    source_directory = os.path.dirname(source_path)
    include_base_directory = os.path.join(
        source_directory, f"include_{source_file_name}"
    )
    if not os.path.isdir(include_base_directory):
        missing_files.append(include_base_directory)
        return missing_files

    # Add original source to includes directory
    shutil.copy2(source_path, include_base_directory)

    # Next gather all includes in each source file
    includes = []
    for dirpath, dirnames, filenames in os.walk(include_base_directory):
        for file in filenames:
            file_source = open(os.path.join(dirpath, file), "r").read()
            includes.extend(produce_include_summary(file_source))

    # Check for missing files
    already_checked = set()
    for include in includes:
        if include in already_checked:
            continue
        if not os.path.isfile(os.path.join(include_base_directory, include)):
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
            log.append(f"Found unsupported idiom matching regex: {line}" + "\n")
            log.append(f"Match was: {match.group(0)}" + "\n")
            log.append(f"Line was: {line_number}" + "\n")
            log.append(f"\n")
    return log


def fix_unsupported_idioms(source: str):
    """
    Preprocesses Fortran source code to convert continuation lines to tree-sitter supported format:
    1. Replaces the first occurrence of '|' with '&' if it is the first non-whitespace character in the line.
    2. Adds an additional '&' to the previous line
    """
    processed_lines = []
    for i, line in enumerate(source.splitlines()):
        if line.lstrip().startswith("|"):
            line = line.replace("|", "&", 1)
            processed_lines[-1] += "&"
        processed_lines.append(line)
    source = "\n".join(processed_lines)

    return source


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


def run_c_preprocessor(source: str, include_base_path: str) -> str:
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


def tree_sitter_parse(source: str) -> Tree:
    """Use tree-sitter to parse the source code and output the parse tree"""
    parser = Parser()

    parser.set_language(
        Language(
            os.path.join(os.path.dirname("../"), LANGUAGE_LIBRARY_REL_PATH), "fortran"
        )
    )

    return parser.parse(bytes(source, "utf8"))
