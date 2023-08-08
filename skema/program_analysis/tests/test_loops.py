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

# NOTE: these examples are very trivial for the realm of recursion
#       more complex ones will follow later as needed

def while1():
    return """
x = 2
while x < 5:
    x = x + 1
    """

def while2():
    return """
x = 2
y = 3

while x < 5:
    x = x + y
    """

def while3():
    return """
x = 2
y = 3

while x < 5:
    x = x + 1
    x = x + y
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

def test_while1():
    while_gromet = generate_gromet(while1())

    base_fn = while_gromet.fn 
    predicate_fn = while_gromet.fn_array[1]

    # Base FN with loop

    assert len(base_fn.wfl) == 1
    assert len(base_fn.pil) == 1
    assert len(base_fn.pol) == 1
    assert len(base_fn.bl) == 1

    # Check Ports
    assert base_fn.pof[0].name == "x" and base_fn.pof[0].box == 1
    assert base_fn.pil[0].name == "x" and base_fn.pil[0].name == "x"

    # Check boxes
    assert len(base_fn.bl) == 1 
    assert base_fn.bl[0].condition == 2 and base_fn.bl[0].body == 3

    # Check Wires
    assert len(base_fn.wfl) == 1 
    assert base_fn.wfl[0].src == 1 and base_fn.wfl[0].tgt == 1

    ###############################################
    assert predicate_fn.b[0].function_type == FunctionType.PREDICATE

    # Check port counts
    assert len(predicate_fn.opi) == 1
    assert len(predicate_fn.opo) == 2
    assert len(predicate_fn.wopio) == 1
    assert len(predicate_fn.wfopi) == 1
    assert len(predicate_fn.wff) == 1
    assert len(predicate_fn.wfopo) == 1

    # Check bf count
    assert len(predicate_fn.bf) == 2

    # Check wires
    assert predicate_fn.wopio[0].src == 1 and predicate_fn.wopio[0].tgt == 1
    assert predicate_fn.wfopo[0].src == 2 and predicate_fn.wfopo[0].tgt == 2
    
    # Check bf
    assert predicate_fn.bf[1].name == "ast.Lt" 


def test_while2():
    while_gromet = generate_gromet(while2())
    base_fn = while_gromet.fn 
    predicate_fn = while_gromet.fn_array[2]
    
    assert predicate_fn.b[0].function_type == FunctionType.PREDICATE
    
    # Base fn with loop
    # Check gromet element counts
    assert len(base_fn.bf) == 2
    assert len(base_fn.wfl) == 2
    assert len(base_fn.bl) == 1 

    # Check wires
    assert base_fn.wfl[0].src == 1 and base_fn.wfl[0].tgt == 1
    assert base_fn.wfl[1].src == 2 and base_fn.wfl[1].tgt == 2
    assert base_fn.bl[0].condition == 3 and base_fn.bl[0].body == 4

    # Check ports
    assert base_fn.pil[0].name == "x" and base_fn.pil[1].name == "y"
    assert base_fn.pol[0].name == "x" and base_fn.pol[1].name == "y"

    # Check predicate
    # Check port counts
    assert len(predicate_fn.opi) == 2
    assert len(predicate_fn.opo) == 3
    assert len(predicate_fn.wopio) == 2
    assert len(predicate_fn.wfopi) == 1
    assert len(predicate_fn.wff) == 1
    assert len(predicate_fn.wfopo) == 1

    # Check bf count
    assert len(predicate_fn.bf) == 2

    # Check wires
    assert predicate_fn.wopio[0].src == 1 and predicate_fn.wopio[0].tgt == 1
    assert predicate_fn.wopio[1].src == 2 and predicate_fn.wopio[1].tgt == 2

    assert predicate_fn.wfopi[0].src == 1 and predicate_fn.wfopi[0].tgt == 1
    assert predicate_fn.wff[0].src == 2 and predicate_fn.wff[0].tgt == 1
    assert predicate_fn.wfopo[0].src == 3 and predicate_fn.wfopo[0].tgt == 2
    
    # Check bf
    assert predicate_fn.bf[1].name == "ast.Lt" 


def test_while3():
    while_gromet = generate_gromet(while3())
    base_fn = while_gromet.fn 
    predicate_fn = while_gromet.fn_array[2]

    assert predicate_fn.b[0].function_type == FunctionType.PREDICATE

    # Base fn with loop
    # Check gromet element counts
    assert len(base_fn.bf) == 2
    assert len(base_fn.wfl) == 2
    assert len(base_fn.bl) == 1 

    # Check wires
    assert base_fn.wfl[0].src == 1 and base_fn.wfl[0].tgt == 1
    assert base_fn.wfl[1].src == 2 and base_fn.wfl[1].tgt == 2
    assert base_fn.bl[0].condition == 3 and base_fn.bl[0].body == 4

    # Check ports
    assert base_fn.pil[0].name == "x" and base_fn.pil[1].name == "y"
    assert base_fn.pol[0].name == "x" and base_fn.pol[1].name == "y"


    # Check predicate
    # Check port counts
    assert len(predicate_fn.opi) == 2
    assert len(predicate_fn.opo) == 3
    assert len(predicate_fn.wopio) == 2
    assert len(predicate_fn.wfopi) == 1
    assert len(predicate_fn.wff) == 1
    assert len(predicate_fn.wfopo) == 1
    assert len(predicate_fn.pif) == 2
    assert len(predicate_fn.pof) == 2

    # Check port boxes
    assert predicate_fn.pif[0].box == 2 and predicate_fn.pif[1].box == 2
    assert predicate_fn.pof[0].box == 1 and predicate_fn.pof[1].box == 2

    # Check bf count
    assert len(predicate_fn.bf) == 2

    # Check wires
    assert predicate_fn.wopio[0].src == 1 and predicate_fn.wopio[0].tgt == 1
    assert predicate_fn.wopio[1].src == 2 and predicate_fn.wopio[1].tgt == 2

    assert predicate_fn.wfopi[0].src == 1 and predicate_fn.wfopi[0].tgt == 1
    assert predicate_fn.wff[0].src == 2 and predicate_fn.wff[0].tgt == 1
    assert predicate_fn.wfopo[0].src == 3 and predicate_fn.wfopo[0].tgt == 2
    
    # Check bf
    assert predicate_fn.bf[1].name == "ast.Lt" 



def test_conditional():
    test_while1()
    test_while2()
    test_while3()
