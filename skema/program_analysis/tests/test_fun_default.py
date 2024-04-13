# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import (
    GrometFNModuleCollection,
    FunctionType,
    ImportType,
    TypedValue
)
import ast

from skema.program_analysis.CAST.pythonAST import py_ast_to_cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

def fun_default1():
    return """
def foo(x=1, y=2):
    return x + y

p = foo()
q = foo(y=5)
r = foo(4)
s = foo(3, 4)
"""

def fun_default2():
    return """
def foo(x=1, y=2):
    return x + y

a = foo(10, y=20)
b = foo(y=2, x=1)
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

def test_fun_default1():
    exp_gromet = generate_gromet(fun_default1())
    base_fn = exp_gromet.fn

    assert len(base_fn.b) == 1
    assert base_fn.b[0].function_type == FunctionType.MODULE

    assert len(base_fn.bf) == 8
    assert base_fn.bf[0].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[0].import_type == ImportType.OTHER 
    assert base_fn.bf[0].body == 1 and base_fn.bf[0].name == "foo_id0"
    assert base_fn.bf[1].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[1].import_type == ImportType.OTHER 
    assert base_fn.bf[1].body == 1 and base_fn.bf[1].name == "foo_id0"
    assert base_fn.bf[2].function_type == FunctionType.EXPRESSION and base_fn.bf[2].body == 2
    assert base_fn.bf[3].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[3].import_type == ImportType.OTHER
    assert base_fn.bf[3].body == 1 and base_fn.bf[3].name == "foo_id0"

    assert base_fn.bf[4].function_type == FunctionType.LITERAL and base_fn.bf[4].value.value == 4
    assert base_fn.bf[4].value.value_type == "Integer"
    assert base_fn.bf[5].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[5].import_type == ImportType.OTHER
    assert base_fn.bf[5].body == 1 and base_fn.bf[5].name == "foo_id0"
    assert base_fn.bf[6].function_type == FunctionType.LITERAL and base_fn.bf[6].value.value == 3
    assert base_fn.bf[6].value.value_type == "Integer"
    assert base_fn.bf[7].function_type == FunctionType.LITERAL and base_fn.bf[7].value.value == 4
    assert base_fn.bf[7].value.value_type == "Integer"

    assert len(base_fn.pif) == 8
    assert base_fn.pif[0].box == 1
    assert base_fn.pif[1].box == 1
    assert base_fn.pif[2].box == 2
    assert base_fn.pif[3].box == 2
    assert base_fn.pif[4].box == 4
    assert base_fn.pif[5].box == 4
    assert base_fn.pif[6].box == 6
    assert base_fn.pif[7].box == 6

    assert len(base_fn.pof) == 8
    assert base_fn.pof[0].box == 1
    assert base_fn.pof[0].name == "p"

    assert base_fn.pof[1].box == 3
    assert base_fn.pof[1].name == "y"

    assert base_fn.pof[2].box == 2
    assert base_fn.pof[2].name == "q"

    assert base_fn.pof[3].box == 5

    assert base_fn.pof[4].box == 4
    assert base_fn.pof[4].name == "r"
    
    assert base_fn.pof[5].box == 7
    assert base_fn.pof[6].box == 8
    
    assert base_fn.pof[7].box == 6
    assert base_fn.pof[7].name == "s"

    assert len(base_fn.wff) == 4
    assert base_fn.wff[0].src == 4
    assert base_fn.wff[0].tgt == 2

    assert base_fn.wff[1].src == 5
    assert base_fn.wff[1].tgt == 4

    assert base_fn.wff[2].src == 7
    assert base_fn.wff[2].tgt == 6

    assert base_fn.wff[3].src == 8
    assert base_fn.wff[3].tgt == 7

    
    ###############################
    func_fn = exp_gromet.fn_array[0]
    assert len(func_fn.b) == 1
    assert func_fn.b[0].function_type == FunctionType.FUNCTION
    assert func_fn.b[0].name == "foo_id0"

    assert len(func_fn.opi) == 2
    assert func_fn.opi[0].box == 1
    assert func_fn.opi[0].name == "x"
    assert func_fn.opi[0].default_value == 1

    assert func_fn.opi[1].box == 1
    assert func_fn.opi[1].name == "y"
    assert func_fn.opi[1].default_value == 2

    assert len(func_fn.opo) == 1
    assert func_fn.opo[0].box == 1

    assert len(func_fn.bf) == 1
    assert func_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert func_fn.bf[0].name == "ast.Add"

    assert len(func_fn.pif) == 2
    assert func_fn.pif[0].box == 1
    assert func_fn.pif[1].box == 1

    assert len(func_fn.pof) == 1
    assert func_fn.pof[0].box == 1

    assert len(func_fn.wfopi) == 2
    assert func_fn.wfopi[0].src == 1
    assert func_fn.wfopi[0].tgt == 1
    assert func_fn.wfopi[1].src == 2
    assert func_fn.wfopi[1].tgt == 2

    assert len(func_fn.wfopo) == 1
    assert func_fn.wfopo[0].src == 1
    assert func_fn.wfopo[0].tgt == 1

    ###############################
    assign_fn = exp_gromet.fn_array[1]
    assert len(assign_fn.opo) == 1
    assert assign_fn.opo[0].box == 1

    assert len(assign_fn.b) == 1
    assert assign_fn.b[0].function_type == FunctionType.EXPRESSION

    assert len(assign_fn.bf) == 1
    assert assign_fn.bf[0].function_type == FunctionType.LITERAL
    assert assign_fn.bf[0].value.value_type == "Integer"
    assert assign_fn.bf[0].value.value == 5
    
    assert len(assign_fn.pof) == 1
    assert assign_fn.pof[0].box == 1
    
    assert len(assign_fn.wfopo) == 1
    assert assign_fn.wfopo[0].src == 1
    assert assign_fn.wfopo[0].tgt == 1

def test_fun_default2():
    exp_gromet = generate_gromet(fun_default2())
    base_fn = exp_gromet.fn

    assert len(base_fn.b) == 1
    assert base_fn.b[0].function_type == FunctionType.MODULE

    assert len(base_fn.bf) == 6
    assert base_fn.bf[0].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[0].import_type == ImportType.OTHER 
    assert base_fn.bf[0].body == 1 and base_fn.bf[0].name == "foo_id0"
    assert base_fn.bf[1].function_type == FunctionType.LITERAL and base_fn.bf[1].value.value == 10
    assert base_fn.bf[1].value.value_type == "Integer"
    assert base_fn.bf[2].function_type == FunctionType.EXPRESSION and base_fn.bf[2].body == 2
    assert base_fn.bf[3].function_type == FunctionType.IMPORTED_METHOD and base_fn.bf[3].import_type == ImportType.OTHER
    assert base_fn.bf[3].body == 1 and base_fn.bf[3].name == "foo_id0"
    assert base_fn.bf[4].function_type == FunctionType.EXPRESSION and base_fn.bf[4].body == 3
    assert base_fn.bf[5].function_type == FunctionType.EXPRESSION and base_fn.bf[5].body == 4

    assert len(base_fn.pif) == 4
    assert base_fn.pif[0].box == 1
    assert base_fn.pif[1].box == 1
    assert base_fn.pif[2].box == 4
    assert base_fn.pif[3].box == 4

    assert len(base_fn.pof) == 6
    assert base_fn.pof[0].box == 2

    assert base_fn.pof[1].box == 3
    assert base_fn.pof[1].name == "y"

    assert base_fn.pof[2].box == 1
    assert base_fn.pof[2].name == "a"

    assert base_fn.pof[3].box == 5
    assert base_fn.pof[3].name == "y"

    assert base_fn.pof[4].box == 6
    assert base_fn.pof[4].name == "x"
    
    assert base_fn.pof[5].box == 4
    assert base_fn.pof[5].name == "b"

    assert len(base_fn.wff) == 4
    assert base_fn.wff[0].src == 1
    assert base_fn.wff[0].tgt == 1

    assert base_fn.wff[1].src == 2
    assert base_fn.wff[1].tgt == 2

    assert base_fn.wff[2].src == 4
    assert base_fn.wff[2].tgt == 4

    assert base_fn.wff[3].src == 3
    assert base_fn.wff[3].tgt == 5
    
    ###############################
    func_fn = exp_gromet.fn_array[0]
    assert len(func_fn.b) == 1
    assert func_fn.b[0].function_type == FunctionType.FUNCTION
    assert func_fn.b[0].name == "foo_id0"

    assert len(func_fn.opi) == 2
    assert func_fn.opi[0].box == 1
    assert func_fn.opi[0].name == "x"
    assert func_fn.opi[0].default_value == 1

    assert func_fn.opi[1].box == 1
    assert func_fn.opi[1].name == "y"
    assert func_fn.opi[1].default_value == 2

    assert len(func_fn.opo) == 1
    assert func_fn.opo[0].box == 1

    assert len(func_fn.bf) == 1
    assert func_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert func_fn.bf[0].name == "ast.Add"

    assert len(func_fn.pif) == 2
    assert func_fn.pif[0].box == 1
    assert func_fn.pif[1].box == 1

    assert len(func_fn.pof) == 1
    assert func_fn.pof[0].box == 1

    assert len(func_fn.wfopi) == 2
    assert func_fn.wfopi[0].src == 1
    assert func_fn.wfopi[0].tgt == 1
    assert func_fn.wfopi[1].src == 2
    assert func_fn.wfopi[1].tgt == 2

    assert len(func_fn.wfopo) == 1
    assert func_fn.wfopo[0].src == 1
    assert func_fn.wfopo[0].tgt == 1

    ###############################
    assign_fn = exp_gromet.fn_array[1]
    assert len(assign_fn.opo) == 1
    assert assign_fn.opo[0].box == 1

    assert len(assign_fn.b) == 1
    assert assign_fn.b[0].function_type == FunctionType.EXPRESSION

    assert len(assign_fn.bf) == 1
    assert assign_fn.bf[0].function_type == FunctionType.LITERAL
    assert assign_fn.bf[0].value.value_type == "Integer"
    assert assign_fn.bf[0].value.value == 20
    
    assert len(assign_fn.pof) == 1
    assert assign_fn.pof[0].box == 1
    
    assert len(assign_fn.wfopo) == 1
    assert assign_fn.wfopo[0].src == 1
    assert assign_fn.wfopo[0].tgt == 1

    ###############################
    assign_fn = exp_gromet.fn_array[2]
    assert len(assign_fn.opo) == 1
    assert assign_fn.opo[0].box == 1

    assert len(assign_fn.b) == 1
    assert assign_fn.b[0].function_type == FunctionType.EXPRESSION

    assert len(assign_fn.bf) == 1
    assert assign_fn.bf[0].function_type == FunctionType.LITERAL
    assert assign_fn.bf[0].value.value_type == "Integer"
    assert assign_fn.bf[0].value.value == 2
    
    assert len(assign_fn.pof) == 1
    assert assign_fn.pof[0].box == 1
    
    assert len(assign_fn.wfopo) == 1
    assert assign_fn.wfopo[0].src == 1
    assert assign_fn.wfopo[0].tgt == 1

    ###############################
    assign_fn = exp_gromet.fn_array[3]
    assert len(assign_fn.opo) == 1
    assert assign_fn.opo[0].box == 1

    assert len(assign_fn.b) == 1
    assert assign_fn.b[0].function_type == FunctionType.EXPRESSION

    assert len(assign_fn.bf) == 1
    assert assign_fn.bf[0].function_type == FunctionType.LITERAL
    assert assign_fn.bf[0].value.value_type == "Integer"
    assert assign_fn.bf[0].value.value == 1
    
    assert len(assign_fn.pof) == 1
    assert assign_fn.pof[0].box == 1
    
    assert len(assign_fn.wfopo) == 1
    assert assign_fn.wfopo[0].src == 1
    assert assign_fn.wfopo[0].tgt == 1

