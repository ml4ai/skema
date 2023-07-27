# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection
import ast

from skema.program_analysis.PyAST2CAST import py_ast_to_cast
from skema.program_analysis.CAST.Fortran.model.cast import SourceRef
from skema.program_analysis.CAST.Fortran import cast
from skema.program_analysis.CAST.Fortran.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

# NOTE: these examples are very trivial for the realm of recursion
#       more complex ones will follow later as needed

def exp0():
    return """
x = 2
    """

def exp1():
    return """
x = 2 + 3
    """

def exp2():
    return """
x = 2
y = x + 3
    """

def exp3():
    return """
x = 2
y = x
    """

def exp4():
    return """
def exp(a,b,c,d):
    s_n = (-a * b * c) + d
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

def test_exp0():
    exp_gromet = generate_gromet(exp0())
    
    base_fn = exp_gromet.fn

    assert len(base_fn.bf) == 1
    assert len(base_fn.pof) == 1

    assert base_fn.bf[0].body == 1
    assert base_fn.pof[0].name == "x"

    ##############################################
    inner_fn = exp_gromet.fn_array[0]

    # Check length of elements
    assert len(inner_fn.opo) == 1
    assert len(inner_fn.bf) == 1
    assert len(inner_fn.pof) == 1
    assert len(inner_fn.wfopo) == 1

    # Check wiring
    assert inner_fn.wfopo[0].src == 1 and inner_fn.wfopo[0].tgt == 1

def test_exp1():
    exp_gromet = generate_gromet(exp1())
    
    inner_fn = exp_gromet.fn_array[0]

    # Check length of elements
    assert len(inner_fn.opo) == 1
    assert len(inner_fn.bf) == 3
    assert len(inner_fn.wff) == 2
    assert len(inner_fn.wfopo) == 1

    # Check wiring
    assert inner_fn.wff[0].src == 1 and inner_fn.wff[0].tgt == 1
    assert inner_fn.wff[1].src == 2 and inner_fn.wff[1].tgt == 2
    assert inner_fn.wfopo[0].src == 1 and inner_fn.wfopo[0].tgt == 3


def test_exp2():
    exp_gromet = generate_gromet(exp2())

    base_fn = exp_gromet.fn

    assert len(base_fn.bf) == 2
    assert len(base_fn.pof) == 2
    assert base_fn.wff[0].src == 1 and base_fn.wff[0].tgt == 1

    # Check calls
    assert base_fn.bf[0].body == 1
    assert base_fn.bf[1].body == 2


    ##############################################
    inner_fn = exp_gromet.fn_array[0]

    # Check length of elements
    assert len(inner_fn.opo) == 1
    assert len(inner_fn.bf) == 1
    assert len(inner_fn.pof) == 1
    assert len(inner_fn.wfopo) == 1

    # Check wiring
    assert inner_fn.wfopo[0].src == 1 and inner_fn.wfopo[0].tgt == 1
    
    ##############################################
    outer_fn = exp_gromet.fn_array[1]

    # Check length of elements
    assert len(outer_fn.opi) == 1
    assert len(outer_fn.opo) == 1
    assert len(outer_fn.bf) == 2
    assert len(outer_fn.pif) == 2
    assert len(outer_fn.pof) == 2
    assert len(outer_fn.wfopi) == 1
    assert len(outer_fn.wff) == 1
    assert len(outer_fn.wfopo) == 1

    # Check wiring
    assert outer_fn.wfopi[0].src == 1 and outer_fn.wfopi[0].tgt == 1
    assert outer_fn.wff[0].src == 2 and outer_fn.wff[0].tgt == 1
    assert outer_fn.wfopo[0].src == 1 and outer_fn.wfopo[0].tgt == 2


def test_exp3():
    exp_gromet = generate_gromet(exp3())
    base_fn = exp_gromet.fn

    assert len(base_fn.bf) == 2
    assert len(base_fn.pif) == 1
    assert len(base_fn.pof) == 2
    assert base_fn.wff[0].src == 1 and base_fn.wff[0].tgt == 1

    # Check calls
    assert base_fn.bf[0].body == 1
    assert base_fn.bf[1].body == 2

    ##############################################
    inner_fn = exp_gromet.fn_array[0]

    # Check length of elements
    assert len(inner_fn.opo) == 1
    assert len(inner_fn.bf) == 1
    assert len(inner_fn.pof) == 1
    assert len(inner_fn.wfopo) == 1

    # Check wiring
    assert inner_fn.wfopo[0].src == 1 and inner_fn.wfopo[0].tgt == 1

    ##############################################
    outer_fn = exp_gromet.fn_array[1]

    # Check length of elements
    assert len(outer_fn.opi) == 1
    assert len(outer_fn.opo) == 1
    assert len(outer_fn.wopio) == 1

    # Check wiring
    assert outer_fn.wopio[0].src == 1 and outer_fn.wopio[0].tgt == 1


def test_exp4():
    exp_gromet = generate_gromet(exp4())

    # FN for outer box
    outer_fn = exp_gromet.fn_array[0]
    assert len(outer_fn.opi) == 4

    # Check wiring to inner expression is correct
    assert outer_fn.wfopi[0].src == 1 and outer_fn.wfopi[0].tgt == 1
    assert outer_fn.wfopi[1].src == 2 and outer_fn.wfopi[1].tgt == 2
    assert outer_fn.wfopi[2].src == 3 and outer_fn.wfopi[2].tgt == 3
    assert outer_fn.wfopi[3].src == 4 and outer_fn.wfopi[3].tgt == 4

    ##############################################
    inner_fn = exp_gromet.fn_array[1]

    # Check length of elements
    assert len(inner_fn.opi) == 4
    assert len(inner_fn.opo) == 1
    assert len(inner_fn.bf) == 4
    assert len(inner_fn.wfopi) == 4
    assert len(inner_fn.wff) == 3
    assert len(inner_fn.wfopo) == 1

    # Check wiring
    assert inner_fn.wfopi[0].src == 1 and inner_fn.wfopi[0].tgt == 1
    assert inner_fn.wfopi[1].src == 3 and inner_fn.wfopi[1].tgt == 2
    assert inner_fn.wfopi[2].src == 5 and inner_fn.wfopi[2].tgt == 3
    assert inner_fn.wfopi[3].src == 7 and inner_fn.wfopi[3].tgt == 4

    assert inner_fn.wff[0].src == 2 and inner_fn.wff[0].tgt == 1
    assert inner_fn.wff[1].src == 4 and inner_fn.wff[1].tgt == 2
    assert inner_fn.wff[2].src == 6 and inner_fn.wff[2].tgt == 3

    assert inner_fn.wfopo[0].src == 1 and inner_fn.wfopo[0].tgt == 4


def test_expression():
    test_exp0()
    test_exp1()
    test_exp2()
    test_exp3()
    test_exp4()

    return