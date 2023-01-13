"""
This program reads a JSON file that contains the CAST representation
of a program, and transforms it to annotated CAST. It then calls a
series of passes that each augment the information in the annotatd CAST nodes
in preparation for the GrFN generation.

One command-line argument is expected, namely the name of the JSON file that
contains the CAST data.
"""

import sys
import dill
import argparse

from skema.utils.script_functions import ann_cast_pipeline
from skema.utils.fold import dictionary_to_gromet_json, del_nulls


def get_args():
    parser = argparse.ArgumentParser(
        description="Runs Annotated Cast pipeline on input CAST json file."
    )
    parser.add_argument(
        "--grfn_2_2",
        help="Generate GrFN 2.2 for the CAST-> Annotated Cast  -> GrFN pipeline",
        action="store_true",
    )
    parser.add_argument(
        "--gromet",
        help="Generates GroMEt using the AnnCAST. CAST -> AnnCast -> GroMEt",
        action="store_true",
    )
    parser.add_argument(
        "--agraph",
        help="Generates a pdf of the Annotated CAST",
        action="store_true",
    )
    parser.add_argument("cast_json", help="input CAST.json file")
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    ann_cast_pipeline(
        args.cast_json,
        gromet=args.gromet,
        grfn_2_2=args.grfn_2_2,
        a_graph=args.agraph,
        from_obj=False,
        indent_level=2,
    )
