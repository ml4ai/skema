import os
import sys
import ast
from skema.program_analysis import astpp
import json
import argparse

from skema.program_analysis.CAST.pythonAST import py_ast_to_cast
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.CAST2FN.visitors.cast_to_agraph_visitor import (
    CASTToAGraphVisitor,
)

from skema.program_analysis.CAST.python.pts2cast import PyTS2CAST

from typing import Optional


def get_args():
    parser = argparse.ArgumentParser(
        "Runs Python to CAST pipeline on input Python source file."
    )
    parser.add_argument(
        "--astpp", help="Dumps Python AST to stdout", action="store_true"
    )
    parser.add_argument(
        "--rawjson",
        help="Dumps out raw JSON contents to stdout",
        action="store_true",
    )
    parser.add_argument(
        "--stdout",
        help="Dumps CAST JSON to stdout instead of a file",
        action="store_true",
    )
    parser.add_argument(
        "--agraph",
        help="Generates visualization of CAST as a PDF file",
        action="store_true",
    )
    parser.add_argument(
        "--legacy",
        help="Generate CAST for GrFN 2.2 pipeline",
        action="store_true",
    )
    parser.add_argument(
        "--ts",
        help="Generate CAST using tree-sitter parser generator instead of the Python AST",
        action="store_true",
    )
    parser.add_argument("pyfile_path", help="input Python source file")
    options = parser.parse_args()
    return options


def python_to_cast(
    pyfile_path,
    agraph=False,
    astprint=False,
    std_out=False,
    rawjson=False,
    legacy=False,
    cast_obj=False,
    tree_sitter=False
) -> Optional[CAST]:
    """Create a CAST object from a Python file and serialize it to JSON.

    Args:
        pyfile_path: Path to the Python source file
        agraph: If true, a PDF visualization of the graph is created.
        astprint: View the AST using the astpp module.
        std_out: If true, the CAST JSON is printed to stdout instead
                 of written to a file.
        rawjson: If true, the raw JSON contents are printed to stdout.
        legacy: If true, generate CAST for GrFN 2.2 pipeline.
        cast_obj: If true, returns the CAST as an object instead of printing to
                stdout.

    Returns:
        If cast_obj is set to True, returns the CAST as an object. Else,
        returns None.
    """

    if not tree_sitter:
        # Open Python file as a giant string
        with open(pyfile_path) as f:
            file_contents = f.read()

        file_name = pyfile_path.split("/")[-1]

        # Count the number of lines in the file
        with open(pyfile_path) as f:
            file_list = f.readlines()
            line_count = 0
            for l in file_list:
                line_count += 1

        # Create a PyASTToCAST Object
        if legacy:
            convert = py_ast_to_cast.PyASTToCAST(file_name, legacy=True)
        else:
            convert = py_ast_to_cast.PyASTToCAST(file_name)

        # Additional option to allow us to view the PyAST
        # using the astpp module
        if astprint:
            astpp.parseprint(file_contents)

        # 'Root' the current working directory so that it's where the
        # Source file we're generating CAST for is (for Import statements)
        old_path = os.getcwd()
        try:
            idx = pyfile_path.rfind("/")

            if idx > -1:
                curr_path = pyfile_path[0:idx]
                os.chdir(curr_path)
            else:
                curr_path = "./" + pyfile_path

            # Parse the Python program's AST and create the CAST
            contents = ast.parse(file_contents)
            C = convert.visit(contents, {}, {})
            C.source_refs = [SourceRef(file_name, None, None, 1, line_count)]
        finally:
            os.chdir(old_path)

        out_cast = cast.CAST([C], "python")
    else:
        file_name = pyfile_path.split("/")[-1]
        out_cast = PyTS2CAST(pyfile_path).out_cast

    if agraph:
        V = CASTToAGraphVisitor(out_cast)
        last_slash_idx = file_name.rfind("/")
        file_ending_idx = file_name.rfind(".")
        pdf_file_name = (
            f"{file_name[last_slash_idx + 1 : file_ending_idx]}.pdf"
        )
        V.to_pdf(pdf_file_name)

    # Then, print CAST as JSON
    if cast_obj:
        return out_cast
    else:
        if rawjson:
            print(
                json.dumps(
                    out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )
        else:
            if std_out:
                print(out_cast.to_json_str())
            else:
                out_name = file_name.split(".")[0]
                print("Writing CAST to " + out_name + "--CAST.json")
                out_handle = open(out_name + "--CAST.json", "w")
                out_handle.write(out_cast.to_json_str())


if __name__ == "__main__":
    args = get_args()
    python_to_cast(
        args.pyfile_path,
        args.agraph,
        args.astpp,
        args.stdout,
        args.rawjson,
        args.legacy,
        tree_sitter=args.ts
    )
