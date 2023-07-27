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

def fun_nested_def1():
    return """
def bar():  # defines module.bar()
    print("module.bar()")

def foo(): # defines module.foo()
    # NOTE: needs global 'bar' call to run in Python appropriately
    bar()  # calling module.bar()
    def bar(): # defining module.foo.bar()
        print("module.foo.bar()")
    bar() # calling module.foo.bar()

foo()
bar() # calls module.bar()
    """

def fun_nested_def2():
    return """
def foo():
    x = 10
    y = x + 2

    def bar(y):  # translate these inner y's to y-inner1 or something to differentiate local and arguments that share the same name
        print(x)
        print(y)
    bar(y)

foo()
    """

def fun_nested_def3():
    return """
def foo(z):
    x = 10
    y = x + 2

    def bar(y):  # translate these inner y's to y-inner1 or something to differentiate local and arguments that share the same name
        a = x + y
        x += 1

        def baz(a):
            b = a + 1
            z = x + y

        baz(x)

    bar(y)

foo(10)
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

def test_closures():
    prog1_gromet = generate_gromet(fun_nested_def1())
    prog2_gromet = generate_gromet(fun_nested_def2())
    prog3_gromet = generate_gromet(fun_nested_def3())

    return
