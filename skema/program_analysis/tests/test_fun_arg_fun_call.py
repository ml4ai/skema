# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection
from skema.gromet.fn import FunctionType
import ast

from skema.program_analysis.CAST.pythonAST import py_ast_to_cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

def fun_arg_fun_call():
    return """
def foo(x,y,z):
    b = x(10) * 2
    c = b * y(3)
    a = x(z) + y(z)
    return a

def a(f): return f + 1
def b(f): return f + 2

foo(x=a,y=b,z=1)
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

def test_fun_arg_fun_call():
    fun_gromet = generate_gromet(fun_arg_fun_call())
    # Test basic properties of assignment node
    base_fn = fun_gromet.fn

    assert len(base_fn.bf) == 4
    assert base_fn.bf[1].value.value_type == "string"
    assert base_fn.bf[1].value.value == "a"

    assert base_fn.bf[2].value.value_type == "string"
    assert base_fn.bf[2].value.value == "b"

    assert len(base_fn.pif) == 3
    assert len(base_fn.pof) == 3
    assert base_fn.pof[0].name == "x" 
    assert base_fn.pof[0].box == 2
    
    assert base_fn.pof[1].name == "y" 
    assert base_fn.pof[1].box == 3

    assert base_fn.pof[2].name == "z" 
    assert base_fn.pof[2].box == 4

    assert len(base_fn.wff) == 3
    assert base_fn.wff[0].src == 1
    assert base_fn.wff[0].tgt == 1

    assert base_fn.wff[1].src == 2
    assert base_fn.wff[1].tgt == 2

    assert base_fn.wff[2].src == 3
    assert base_fn.wff[2].tgt == 3

    ################################################################## 

    foo_fn = fun_gromet.fn_array[0]
    assert len(foo_fn.opi) == 3
    assert foo_fn.opi[0].name == "x"
    assert foo_fn.opi[1].name == "y"
    assert foo_fn.opi[2].name == "z"

    assert len(foo_fn.opo) == 1
    assert len(foo_fn.bf) == 3

    assert len(foo_fn.pif) == 6
    assert foo_fn.pif[0].box == 1
    assert foo_fn.pif[1].box == 2
    assert foo_fn.pif[2].box == 2
    assert foo_fn.pif[3].box == 3
    assert foo_fn.pif[4].box == 3
    assert foo_fn.pif[5].box == 3

    assert len(foo_fn.pof) == 3
    assert foo_fn.pof[0].box == 1
    assert foo_fn.pof[0].name == "b"

    assert foo_fn.pof[1].box == 2
    assert foo_fn.pof[1].name == "c"

    assert foo_fn.pof[2].box == 3
    assert foo_fn.pof[2].name == "a"

    assert len(foo_fn.wfopi) == 5
    assert foo_fn.wfopi[0].src == 1 and foo_fn.wfopi[0].tgt == 1
    assert foo_fn.wfopi[1].src == 2 and foo_fn.wfopi[1].tgt == 2
    assert foo_fn.wfopi[2].src == 4 and foo_fn.wfopi[2].tgt == 1
    assert foo_fn.wfopi[3].src == 5 and foo_fn.wfopi[3].tgt == 3
    assert foo_fn.wfopi[4].src == 6 and foo_fn.wfopi[4].tgt == 2

    assert len(foo_fn.wff) == 1
    assert foo_fn.wff[0].src == 3 and foo_fn.wff[0].tgt == 1

    assert len(foo_fn.wfopo) == 1
    assert foo_fn.wfopo[0].src == 1 and foo_fn.wfopo[0].tgt == 3


    ################################################################## 
    first_call_fn = fun_gromet.fn_array[1]
    assert len(first_call_fn.opi) == 1
    assert len(first_call_fn.opo) == 1

    assert len(first_call_fn.bf) == 2
    assert first_call_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert first_call_fn.bf[0].name == "_call"

    assert first_call_fn.bf[1].function_type == FunctionType.LITERAL
    assert first_call_fn.bf[1].value.value == 10

    assert len(first_call_fn.pif) == 2
    assert first_call_fn.pif[0].box == 1
    assert first_call_fn.pif[1].box == 1

    assert len(first_call_fn.pof) == 2
    assert first_call_fn.pof[0].box == 1
    assert first_call_fn.pof[1].box == 2

    assert len(first_call_fn.wfopi) == 1
    assert first_call_fn.wfopi[0].src == 1 and first_call_fn.wfopi[0].tgt == 1

    assert len(first_call_fn.wff) == 1
    assert first_call_fn.wff[0].src == 2 and first_call_fn.wff[0].tgt == 2

    assert len(first_call_fn.wfopo) == 1
    assert first_call_fn.wfopo[0].src == 1 and first_call_fn.wfopo[0].tgt == 1

    ################################################################## 
    first_mult_fn = fun_gromet.fn_array[2]
    assert len(first_mult_fn.opi) == 1
    assert len(first_mult_fn.opo) == 1

    assert len(first_mult_fn.bf) == 3
    assert first_mult_fn.bf[0].function_type == FunctionType.EXPRESSION
    assert first_mult_fn.bf[0].body == 2

    assert first_mult_fn.bf[1].function_type == FunctionType.LITERAL
    assert first_mult_fn.bf[1].value.value == 2

    assert first_mult_fn.bf[2].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert first_mult_fn.bf[2].name == "ast.Mult"

    assert len(first_mult_fn.pif) == 3
    assert first_mult_fn.pif[0].box == 1
    assert first_mult_fn.pif[1].box == 3
    assert first_mult_fn.pif[2].box == 3

    assert len(first_mult_fn.pof) == 3
    assert first_mult_fn.pof[0].box == 1
    assert first_mult_fn.pof[1].box == 2
    assert first_mult_fn.pof[2].box == 3

    assert len(first_mult_fn.wfopi) == 1
    assert first_mult_fn.wfopi[0].src == 1 and first_mult_fn.wfopi[0].tgt == 1

    assert len(first_mult_fn.wff) == 2
    assert first_mult_fn.wff[0].src == 2 and first_mult_fn.wff[0].tgt == 1
    assert first_mult_fn.wff[1].src == 3 and first_mult_fn.wff[1].tgt == 2

    assert len(first_mult_fn.wfopo) == 1
    assert first_mult_fn.wfopo[0].src == 1 and first_mult_fn.wfopo[0].tgt == 3
    
    ################################################################## 
    second_call_fn = fun_gromet.fn_array[3]
    assert len(second_call_fn.opi) == 1
    assert len(second_call_fn.opo) == 1

    assert len(second_call_fn.bf) == 2
    assert second_call_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert second_call_fn.bf[0].name == "_call"

    assert second_call_fn.bf[1].function_type == FunctionType.LITERAL
    assert second_call_fn.bf[1].value.value == 3

    assert len(second_call_fn.pif) == 2
    assert second_call_fn.pif[0].box == 1
    assert second_call_fn.pif[1].box == 1

    assert len(second_call_fn.pof) == 2
    assert second_call_fn.pof[0].box == 1
    assert second_call_fn.pof[1].box == 2

    assert len(second_call_fn.wfopi) == 1
    assert second_call_fn.wfopi[0].src == 1 and second_call_fn.wfopi[0].tgt == 1

    assert len(second_call_fn.wff) == 1
    assert second_call_fn.wff[0].src == 2 and second_call_fn.wff[0].tgt == 2

    assert len(second_call_fn.wfopo) == 1
    assert second_call_fn.wfopo[0].src == 1 and second_call_fn.wfopo[0].tgt == 1

    ################################################################## 
    second_mult_fn = fun_gromet.fn_array[4]
    assert len(second_mult_fn.opi) == 2
    assert len(second_mult_fn.opo) == 1

    assert len(second_mult_fn.bf) == 2
    assert second_mult_fn.bf[0].function_type == FunctionType.EXPRESSION
    assert second_mult_fn.bf[0].body == 4

    assert second_mult_fn.bf[1].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert second_mult_fn.bf[1].name == "ast.Mult"

    assert len(second_mult_fn.pif) == 3
    assert second_mult_fn.pif[0].box == 1
    assert second_mult_fn.pif[1].box == 2
    assert second_mult_fn.pif[2].box == 2

    assert len(second_mult_fn.pof) == 2
    assert second_mult_fn.pof[0].box == 1
    assert second_mult_fn.pof[1].box == 2

    assert len(second_mult_fn.wfopi) == 2
    assert second_mult_fn.wfopi[0].src == 1 and second_mult_fn.wfopi[0].tgt == 1
    assert second_mult_fn.wfopi[1].src == 2 and second_mult_fn.wfopi[1].tgt == 2

    assert len(second_mult_fn.wff) == 1
    assert second_mult_fn.wff[0].src == 3 and second_mult_fn.wff[0].tgt == 1

    assert len(second_mult_fn.wfopo) == 1
    assert second_mult_fn.wfopo[0].src == 1 and second_mult_fn.wfopo[0].tgt == 2

    ################################################################## 
    third_call_fn = fun_gromet.fn_array[5]
    assert len(third_call_fn.opi) == 2
    assert len(third_call_fn.opo) == 1

    assert len(third_call_fn.bf) == 1
    assert third_call_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert third_call_fn.bf[0].name == "_call"

    assert len(third_call_fn.pif) == 2
    assert third_call_fn.pif[0].box == 1
    assert third_call_fn.pif[1].box == 1

    assert len(third_call_fn.pof) == 1 
    assert third_call_fn.pof[0].box == 1

    assert len(third_call_fn.wfopi) == 2
    assert third_call_fn.wfopi[0].src == 1 and third_call_fn.wfopi[0].tgt == 1
    assert third_call_fn.wfopi[1].src == 2 and third_call_fn.wfopi[1].tgt == 2

    assert len(third_call_fn.wfopo) == 1
    assert third_call_fn.wfopo[0].src == 1 and third_call_fn.wfopo[0].tgt == 1

    ################################################################## 
    fourth_call_fn = fun_gromet.fn_array[6]
    assert len(fourth_call_fn.opi) == 2
    assert len(fourth_call_fn.opo) == 1

    assert len(fourth_call_fn.bf) == 1
    assert fourth_call_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert fourth_call_fn.bf[0].name == "_call"

    assert len(fourth_call_fn.pif) == 2
    assert fourth_call_fn.pif[0].box == 1
    assert fourth_call_fn.pif[1].box == 1

    assert len(fourth_call_fn.pof) == 1 
    assert fourth_call_fn.pof[0].box == 1

    assert len(fourth_call_fn.wfopi) == 2
    assert fourth_call_fn.wfopi[0].src == 1 and fourth_call_fn.wfopi[0].tgt == 1
    assert fourth_call_fn.wfopi[1].src == 2 and fourth_call_fn.wfopi[1].tgt == 2

    assert len(fourth_call_fn.wfopo) == 1
    assert fourth_call_fn.wfopo[0].src == 1 and fourth_call_fn.wfopo[0].tgt == 1

    ################################################################## 
    double_call_fn = fun_gromet.fn_array[7]
    assert len(double_call_fn.opi) == 3
    assert len(double_call_fn.opo) == 1
    assert len(double_call_fn.bf) == 3
    assert double_call_fn.bf[2].function_type == FunctionType.LANGUAGE_PRIMITIVE
    
    assert len(double_call_fn.pif) == 6
    assert double_call_fn.pif[0].box == 1
    assert double_call_fn.pif[1].box == 1
    assert double_call_fn.pif[2].box == 2
    assert double_call_fn.pif[3].box == 2
    assert double_call_fn.pif[4].box == 3
    assert double_call_fn.pif[5].box == 3

    assert len(double_call_fn.pof) == 3
    assert double_call_fn.pof[0].box == 1
    assert double_call_fn.pof[1].box == 2
    assert double_call_fn.pof[2].box == 3

    assert len(double_call_fn.wfopi) == 4
    assert double_call_fn.wfopi[0].src == 1 and double_call_fn.wfopi[0].tgt == 1
    assert double_call_fn.wfopi[1].src == 2 and double_call_fn.wfopi[1].tgt == 2
    assert double_call_fn.wfopi[2].src == 3 and double_call_fn.wfopi[2].tgt == 3
    assert double_call_fn.wfopi[3].src == 4 and double_call_fn.wfopi[3].tgt == 2

    assert len(double_call_fn.wff) == 2
    assert double_call_fn.wff[0].src == 5 and double_call_fn.wff[0].tgt == 1
    assert double_call_fn.wff[1].src == 6 and double_call_fn.wff[1].tgt == 2

    assert len(double_call_fn.wfopo) == 1
    assert double_call_fn.wfopo[0].src == 1 and double_call_fn.wfopo[0].tgt == 3


