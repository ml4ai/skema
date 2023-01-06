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

import CAST2GrFN

from CAST2GrFN.ann_cast.cast_to_annotated_cast import (
    CastToAnnotatedCastVisitor,
)
from CAST2GrFN.cast import CAST
from CAST2GrFN.visitors.cast_to_agraph_visitor import (
    CASTToAGraphVisitor,
)
from CAST2GrFN.ann_cast.id_collapse_pass import (
    IdCollapsePass,
)
from CAST2GrFN.ann_cast.container_scope_pass import (
    ContainerScopePass,
)
from CAST2GrFN.ann_cast.variable_version_pass import (
    VariableVersionPass,
)
from CAST2GrFN.ann_cast.grfn_var_creation_pass import (
    GrfnVarCreationPass,
)
from CAST2GrFN.ann_cast.grfn_assignment_pass import (
    GrfnAssignmentPass,
)
from CAST2GrFN.ann_cast.lambda_expression_pass import (
    LambdaExpressionPass,
)
from CAST2GrFN.ann_cast.to_grfn_pass import (
    ToGrfnPass,
)
from CAST2GrFN.ann_cast.to_gromet_pass import (
    ToGrometPass,
)

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
