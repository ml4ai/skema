# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import (
    GrometFNModuleCollection,
    FunctionType,
    ImportType
)
import ast

from skema.program_analysis.CAST.pythonAST import py_ast_to_cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

def function_type_1():
    return """
import numpy as np

x = np.linspace(0, 1, 10)
"""

def generate_gromet(test_file_string):
    # use ast.Parse to get Python AST
    contents = ast.parse(test_file_string)

    # use Python to CAST
    line_count = len(test_file_string.split("\n"))
    convert = py_ast_to_cast.PyASTToCAST("temp")
    C = convert.visit(contents, {}, {})
    C.source_refs = [SourceRef("temp", None, None, 1, line_count)]
    out_cast = cast.CAST([C], "python")

    # use AnnCastPipeline to create GroMEt
    gromet = ann_cast_pipeline(out_cast, gromet=True, to_file=False, from_obj=True)

    return gromet

def test_function_type1():
    exp_gromet = generate_gromet(function_type_1())
    
    base_fn = exp_gromet.fn

    assert len(base_fn.b) == 1
    assert len(base_fn.bf) == 5

    assert len(base_fn.pif) == 3
    assert len(base_fn.pof) == 6
    assert len(base_fn.wff) == 4

    assert base_fn.bf[1].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[1].import_type == ImportType.NATIVE
    
