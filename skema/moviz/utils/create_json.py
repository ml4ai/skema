import json
import ast

from automates.program_analysis.PyAST2CAST import py_ast_to_cast

from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet

from automates.program_analysis.CAST2GrFN import cast
from automates.program_analysis.CAST2GrFN.model.cast import SourceRef
from automates.program_analysis.CAST2GrFN.ann_cast.cast_to_annotated_cast import (
    CastToAnnotatedCastVisitor,
)

from automates.program_analysis.CAST2GrFN.ann_cast.id_collapse_pass import (
    IdCollapsePass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.container_scope_pass import (
    ContainerScopePass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.variable_version_pass import (
    VariableVersionPass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.grfn_var_creation_pass import (
    GrfnVarCreationPass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.grfn_assignment_pass import (
    GrfnAssignmentPass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.lambda_expression_pass import (
    LambdaExpressionPass,
)
from automates.program_analysis.CAST2GrFN.ann_cast.to_gromet_pass import (
    ToGrometPass,
)

from automates.utils.fold import dictionary_to_gromet_json, del_nulls


# PYTHON_SOURCE_FILE = "exp0.py"
# PROGRAM_NAME = PYTHON_SOURCE_FILE.rsplit(".")[0].rsplit("/")[-1]


def run_python_to_cast(PYTHON_SOURCE_FILE, PROGRAM_NAME):
    """run_python_to_cast reads in a Python source file and creates a CAST object and returns it.

    A script that does this and exports it to JSON exists in
    'automates/scripts/program_analysis/python2cast.py'
    """

    # Create the CAST from Python Source
    convert = py_ast_to_cast.PyASTToCAST(PROGRAM_NAME, legacy=False)
    contents = ast.parse(open(PYTHON_SOURCE_FILE).read())
    C = convert.visit(contents, {}, {})
    C.source_refs = [SourceRef(PROGRAM_NAME, None, None, 1, 1)]

    # Return CAST object
    out_cast = cast.CAST([C], "python")
    return out_cast


def run_cast_to_gromet_pipeline(cast):
    """run_cast_to_gromet_pipeline converts a CAST object (generated by run_python_to_cast in this notebook)
    to an AnnotatedCAST (AnnCAST) object. It then runs seven passes over this AnnCAST object to generate
    a GroMEt object which is then returned.

    A script that does this and exports to JSON exists in
    'automates/scripts/program_analysis/run_ann_cast_pipeline.py'

    The individual AnnCAST passes exist under
    'automates/program_analysis/CAST2GrFN/ann_cast/'
    """

    visitor = CastToAnnotatedCastVisitor(cast)
    pipeline_state = visitor.generate_annotated_cast()

    IdCollapsePass(pipeline_state)
    ContainerScopePass(pipeline_state)
    VariableVersionPass(pipeline_state)
    GrfnVarCreationPass(pipeline_state)
    GrfnAssignmentPass(pipeline_state)
    LambdaExpressionPass(pipeline_state)
    ToGrometPass(pipeline_state)

    gromet_object = pipeline_state.gromet_collection

    return gromet_object


def run_pipeline_export_gromet(PYTHON_SOURCE_FILE, PROGRAM_NAME):
    """Runs the two functions in the previous cells to generate CAST and then generate GroMEt
    It then serializes the GroMEt and exports it as a JSON file.

    This uses utilities found in
    'automates/utils/fold.py'
    to serialize the GroMEt
    """
    cast = run_python_to_cast(PYTHON_SOURCE_FILE, PROGRAM_NAME)
    gromet_object = run_cast_to_gromet_pipeline(cast)

    with open(f"{PROGRAM_NAME}--Gromet-FN-auto.json", "w") as f:
        gromet_collection_dict = gromet_object.to_dict()
        f.write(dictionary_to_gromet_json(del_nulls(gromet_collection_dict)))


def import_gromet(PROGRAM_NAME):
    """Reads in a GroMEt JSON file and creates a GroMEt object out of it
    This uses utilities found in
    'automates/program_analysis/JSON2GroMEt.py'
    to import the GroMEt
    """
    gromet_obj = json_to_gromet(f"{PROGRAM_NAME}--Gromet-FN-auto.json")
    return gromet_obj
