# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import (
    GrometFNModuleCollection,
    FunctionType,
    TypedValue,
)
import ast

from skema.program_analysis.CAST.pythonAST import py_ast_to_cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline


def return1():
    return """
def return_true():
    return True
    """

def return2():
    return """
def return_true():
    return True

while (return_true()):
    print("Test")
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

def test_return1():
    gromet = generate_gromet(return1())
    
    base_fn = gromet.fn

    assert len(base_fn.b) == 1

    func_fn = gromet.fn_array[0]
    assert len(func_fn.b) == 1

    assert len(func_fn.opo) == 1
    assert func_fn.opo[0].box == 1

    assert len(func_fn.bf) == 1
    assert func_fn.bf[0].function_type == FunctionType.LITERAL
    assert func_fn.bf[0].value.value_type == "Boolean"
    assert func_fn.bf[0].value.value == "True"

    assert len(func_fn.pof) == 1
    assert func_fn.pof[0].box == 1

    assert len(func_fn.wfopo) == 1
    assert func_fn.wfopo[0].src == 1 and func_fn.wfopo[0].tgt == 1 


def test_return2():
    exp_gromet = generate_gromet(return2())
    
    base_fn = exp_gromet.fn
    assert len(base_fn.bl) == 1
    assert base_fn.bl[0].condition == 2
    assert base_fn.bl[0].body == 3

    func_fn = exp_gromet.fn_array[0]
    assert len(func_fn.b) == 1

    assert len(func_fn.opo) == 1
    assert func_fn.opo[0].box == 1

    assert len(func_fn.bf) == 1
    assert func_fn.bf[0].function_type == FunctionType.LITERAL
    assert func_fn.bf[0].value.value_type == "Boolean"
    assert func_fn.bf[0].value.value == "True"

    assert len(func_fn.pof) == 1
    assert func_fn.pof[0].box == 1

    assert len(func_fn.wfopo) == 1
    assert func_fn.wfopo[0].src == 1 and func_fn.wfopo[0].tgt == 1 

    predicate_fn = exp_gromet.fn_array[1]
    assert len(predicate_fn.b) == 1
    assert len(predicate_fn.opo) == 1
    assert predicate_fn.opo[0].box == 1

    assert len(predicate_fn.bf) == 1
    assert predicate_fn.bf[0].body == 1

    assert len(predicate_fn.pof) == 1
    assert predicate_fn.pof[0].box == 1

    assert len(predicate_fn.wfopo) == 1
    assert predicate_fn.wfopo[0].src == 1
    assert predicate_fn.wfopo[0].tgt == 1

    loop_fn = exp_gromet.fn_array[2]
    assert len(loop_fn.bf) == 1
    assert loop_fn.bf[0].body == 4

    loop_body_fn = exp_gromet.fn_array[3]
    assert len(loop_body_fn.opo) == 1
    assert loop_body_fn.opo[0].box == 1

    assert len(loop_body_fn.bf) == 2
    assert loop_body_fn.bf[1].function_type == FunctionType.LITERAL
    assert loop_body_fn.bf[1].value.value_type == "List"

    assert len(loop_body_fn.pif) == 1
    assert loop_body_fn.pif[0].box == 1

    assert len(loop_body_fn.pof) == 2
    assert loop_body_fn.pof[0].box == 1
    assert loop_body_fn.pof[1].box == 2

    assert len(loop_body_fn.wff) == 1
    assert loop_body_fn.wff[0].src == 1
    assert loop_body_fn.wff[0].tgt == 2

    assert len(loop_body_fn.wfopo) == 1
    assert loop_body_fn.wfopo[0].src == 1
    assert loop_body_fn.wfopo[0].tgt == 1
    