# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Attribute,
    Var,
    Call,
    Name,
    CASTLiteralValue,
    ModelImport,
    StructureType,
)

def import1():
    return """
import sys
    
sys.exit()
    """

def import2():
    return """
import sys as system

system.exit()
    """

def import3():
    return """
from sys import *
    
exit()
    """

def import4():
    return """
import sys
import numpy

x = numpy.array([1,2,3])
sys.exit()
    """

def import5():
    return """
import sys as system, numpy as np

x = np.array([1,2,3])
system.exit()    
    """

def import6():
    return """
from sys import exit, copyright

copyright()
exit()
    """


def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_import1():
    cast = generate_cast(import1())

    import_node = cast.nodes[0].body[0]
    exit_node = cast.nodes[0].body[1]

    assert isinstance(import_node, ModelImport)
    assert import_node.name == "sys"
    assert import_node.alias == None
    assert import_node.symbol == None
    assert import_node.all == False

    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0

    func_call = exit_node.func
    assert isinstance(func_call, Attribute)
    assert isinstance(func_call.attr, Name)
    assert func_call.attr.name == "exit"
    assert func_call.attr.id == 1

    assert isinstance(func_call.value, Name)
    assert func_call.value.name == "sys"
    assert func_call.value.id == 0

def test_import2():
    cast = generate_cast(import2())

    import_node = cast.nodes[0].body[0]
    exit_node = cast.nodes[0].body[1]

    assert isinstance(import_node, ModelImport)
    assert import_node.name == "sys"
    assert import_node.alias == "system"
    assert import_node.symbol == None
    assert import_node.all == False

    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0

    func_call = exit_node.func
    assert isinstance(func_call, Attribute)
    assert isinstance(func_call.attr, Name)
    assert func_call.attr.name == "exit"
    assert func_call.attr.id == 2 # the ID being 2 and not 1 implies that 'system' is ID 1

    assert isinstance(func_call.value, Name)
    assert func_call.value.name == "sys"
    assert func_call.value.id == 0

def test_import3():
    cast = generate_cast(import3())

    import_node = cast.nodes[0].body[0]
    exit_node = cast.nodes[0].body[1]

    assert isinstance(import_node, ModelImport)
    assert import_node.name == "sys"
    assert import_node.alias == None
    assert import_node.symbol == None
    assert import_node.all == True

    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0

    func_call = exit_node.func
    assert isinstance(func_call, Name)
    assert func_call.name == "exit"
    assert func_call.id == 0 

def test_import4():
    cast = generate_cast(import4())

    import_node = cast.nodes[0].body[0]
    assert isinstance(import_node, ModelImport)
    assert import_node.name == "sys"
    assert import_node.alias == None
    assert import_node.symbol == None
    assert import_node.all == False

    import_node = cast.nodes[0].body[1]
    assert isinstance(import_node, ModelImport)
    assert import_node.name == "numpy"
    assert import_node.alias == None
    assert import_node.symbol == None
    assert import_node.all == False

    assign_node = cast.nodes[0].body[2]
    assert isinstance(assign_node, Assignment)
    var = assign_node.left
    assert isinstance(var, Var)
    assert isinstance(var.val, Name)
    assert var.val.name == "x"
    assert var.val.id == 3
    
    call = assign_node.right
    assert isinstance(call, Call)
    args = call.arguments
    assert len(args) == 1
    arg1 = args[0]
    assert isinstance(arg1, CASTLiteralValue)
    assert arg1.value_type == StructureType.LIST

    attr = call.func
    assert isinstance(attr, Attribute)
    assert isinstance(attr.attr, Name)  
    assert attr.attr.name == "array"
    assert attr.attr.id == 2

    assert isinstance(attr.value, Name)  
    assert attr.value.name == "numpy"
    assert attr.value.id == 1

    exit_node = cast.nodes[0].body[3]
    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0

    func_call = exit_node.func
    assert isinstance(func_call, Attribute)
    assert isinstance(func_call.attr, Name)
    assert func_call.attr.name == "exit"
    assert func_call.attr.id == 4 

    assert isinstance(func_call.value, Name)
    assert func_call.value.name == "sys"
    assert func_call.value.id == 0

def test_import5():
    cast = generate_cast(import5())

    import_node = cast.nodes[0].body[0]
    assert isinstance(import_node, ModelImport)
    assert import_node.name == "sys"
    assert import_node.alias == "system"
    assert import_node.symbol == None
    assert import_node.all == False

    import_node = cast.nodes[0].body[1]
    assert isinstance(import_node, ModelImport)
    assert import_node.name == "numpy"
    assert import_node.alias == "np"
    assert import_node.symbol == None
    assert import_node.all == False

    assign_node = cast.nodes[0].body[2]
    assert isinstance(assign_node, Assignment)
    var = assign_node.left
    assert isinstance(var, Var)
    assert isinstance(var.val, Name)
    assert var.val.name == "x"
    assert var.val.id == 5
    
    call = assign_node.right
    assert isinstance(call, Call)
    args = call.arguments
    assert len(args) == 1
    arg1 = args[0]
    assert isinstance(arg1, CASTLiteralValue)
    assert arg1.value_type == StructureType.LIST

    attr = call.func
    assert isinstance(attr, Attribute)
    assert isinstance(attr.attr, Name)  
    assert attr.attr.name == "array"
    assert attr.attr.id == 4

    assert isinstance(attr.value, Name)  
    assert attr.value.name == "numpy"
    assert attr.value.id == 2

    exit_node = cast.nodes[0].body[3]
    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0

    func_call = exit_node.func
    assert isinstance(func_call, Attribute)
    assert isinstance(func_call.attr, Name)
    assert func_call.attr.name == "exit"
    assert func_call.attr.id == 6 

    assert isinstance(func_call.value, Name)
    assert func_call.value.name == "sys"
    assert func_call.value.id == 0

def test_import6():
    cast = generate_cast(import6())

    import1_node = cast.nodes[0].body[0]
    import2_node = cast.nodes[0].body[1]
    cr_node = cast.nodes[0].body[2]
    exit_node = cast.nodes[0].body[3]

    assert isinstance(import1_node, ModelImport)
    assert import1_node.name == "sys"
    assert import1_node.alias == None
    assert import1_node.symbol == "exit"
    assert import1_node.all == False

    assert isinstance(import2_node, ModelImport)
    assert import2_node.name == "sys"
    assert import2_node.alias == None
    assert import2_node.symbol == "copyright"
    assert import2_node.all == False

    assert isinstance(cr_node, Call)
    assert len(cr_node.arguments) == 0
    # assert isinstance(cr_node.func, Var)
    assert isinstance(cr_node.func, Name)
    assert cr_node.func.name == "copyright"
    assert cr_node.func.id == 1

    assert isinstance(exit_node, Call)
    assert len(exit_node.arguments) == 0
    # assert isinstance(exit_node.func, Var)
    assert isinstance(exit_node.func, Name)
    assert exit_node.func.name == "exit"
    assert exit_node.func.id == 0
