# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection
import ast

from skema.program_analysis.PyAST2CAST import py_ast_to_cast
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

# NOTE: these examples are very trivial for the realm of recursion
#       more complex ones will follow later as needed

def recurse1():
    return """
def rec(x):
    rec(x + 1)

z = rec(12)
    """

def recurse2():
    return """
def rec1(x):
    rec2(x + 1)

def rec2(y):
    rec1(y + 2)

z = rec1(12)
    """

def recurse3():
    return """
def rec(x):
    if x < 10:
        return x
    y = rec(x + 1)
    return y

z = rec(1) 
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

def test_recursion():
    prog1_gromet = generate_gromet(recurse1())
    prog2_gromet = generate_gromet(recurse2())
    prog3_gromet = generate_gromet(recurse3())

    return
