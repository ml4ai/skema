# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    FunctionDef,
    ModelReturn,
    Var,
    Call,
    Name,
    CASTLiteralValue,
)

def list1():
    return """
x = [1,2,3]
y = x[0]
    """

def list2():
    return """
x = [1,[2,3]]
y = x[1][0]
    """

def list3():
    return """
x = [1,2,3,4,5]
y = x[0:3]
z = x[0:3:2]
    """

def list4():
    return """
def foo():
    return 2
    
x = [1,2,3,4,5]
y = x[0:foo()]
    """


def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_list1():
    cast = generate_cast(list1())

    asg_node = cast.nodes[0].body[0]
    index_node = cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "List"

    assert isinstance(index_node, Assignment)
    assert isinstance(index_node.left, Var) 
    assert isinstance(index_node.left.val, Name) 
    assert index_node.left.val.name == "y"
    assert index_node.left.val.id == 2

    index_call = index_node.right
    assert isinstance(index_call, Call)
    assert isinstance(index_call.func, Name)    
    assert index_call.func.name == "_get"
    assert index_call.func.id == 1
    assert len(index_call.arguments) == 2

    assert isinstance(index_call.arguments[0], Name), f"is type{type(index_call.arguments[0])}"
    assert index_call.arguments[0].name == "x"
    assert index_call.arguments[0].id == 0

    assert isinstance(index_call.arguments[1], CASTLiteralValue)
    assert index_call.arguments[1].value_type == "Integer"
    assert index_call.arguments[1].value == "0"


def test_list2():
    cast = generate_cast(list2())

    asg_node = cast.nodes[0].body[0]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "List"
    second_list = asg_node.right.value[1]

    assert isinstance(second_list, CASTLiteralValue)
    assert second_list.value_type == "List"
    assert isinstance(second_list.value[0], CASTLiteralValue)
    assert second_list.value[0].value == "2"

    assert isinstance(second_list.value[1], CASTLiteralValue)
    assert second_list.value[1].value == "3"

    index_node = cast.nodes[0].body[1]
    assert isinstance(index_node, Assignment)
    assert isinstance(index_node.left, Var) 
    assert isinstance(index_node.left.val, Name) 
    assert index_node.left.val.name == "y"
    assert index_node.left.val.id == 2

    index_call = index_node.right
    assert isinstance(index_call, Call)
    assert isinstance(index_call.func, Name)    
    assert index_call.func.name == "_get"
    assert index_call.func.id == 1
    assert len(index_call.arguments) == 2

    arg_call = index_call.arguments[0]
    assert isinstance(arg_call, Call), f"is type{type(index_call.arguments[0])}"
    assert arg_call.func.name == "_get"
    assert arg_call.func.id == 1

    assert len(arg_call.arguments) == 2
    assert isinstance(arg_call.arguments[0], Name)
    assert arg_call.arguments[0].name == "x"
    assert arg_call.arguments[0].id == 0

    assert isinstance(arg_call.arguments[1], CASTLiteralValue)
    assert arg_call.arguments[1].value == "1"

    assert isinstance(index_call.arguments[1], CASTLiteralValue)
    assert index_call.arguments[1].value_type == "Integer"
    assert index_call.arguments[1].value == "0"


def test_list3():
    cast = generate_cast(list3())

    asg_node = cast.nodes[0].body[0]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "List"
    assert len(asg_node.right.value) == 5

    index_node = cast.nodes[0].body[1]
    assert isinstance(index_node, Assignment)
    assert isinstance(index_node.left, Var) 
    assert isinstance(index_node.left.val, Name) 
    assert index_node.left.val.name == "y"
    assert index_node.left.val.id == 2

    index_call = index_node.right
    assert isinstance(index_call, Call)
    assert isinstance(index_call.func, Name)    
    assert index_call.func.name == "_get"
    assert index_call.func.id == 1
    assert len(index_call.arguments) == 2

    slice1 = index_call.arguments[0]
    assert isinstance(slice1, Name)
    assert slice1.name == "x"
    assert slice1.id == 0

    slice2 = index_call.arguments[1]
    assert isinstance(slice2, CASTLiteralValue)
    assert slice2.value_type == "List"
    assert len(slice2.value) == 3 

    assert isinstance(slice2.value[0], CASTLiteralValue)
    assert slice2.value[0].value == "0"

    assert isinstance(slice2.value[1], CASTLiteralValue)
    assert slice2.value[1].value == "3"

    assert isinstance(slice2.value[2], CASTLiteralValue)
    assert slice2.value[2].value == "1"

    second_idx = cast.nodes[0].body[2]
    assert isinstance(second_idx, Assignment)
    assert isinstance(second_idx.left, Var)
    assert second_idx.left.val.name == "z"
    assert second_idx.left.val.id == 3

    second_call = second_idx.right
    assert isinstance(second_call, Call)
    assert isinstance(second_call.func, Name)
    assert second_call.func.name == "_get"
    assert second_call.func.id == 1

    second_args = second_call.arguments
    assert len(second_args) == 2
    assert isinstance(second_args[0], Name) 
    assert second_args[0].name == "x"
    assert second_args[0].id == 0

    idx_args = second_args[1]
    assert isinstance(idx_args, CASTLiteralValue)
    assert idx_args.value_type == "List"
    assert len(idx_args.value) == 3 

    assert isinstance(idx_args.value[0], CASTLiteralValue)
    assert idx_args.value[0].value == "0"

    assert isinstance(idx_args.value[1], CASTLiteralValue)
    assert idx_args.value[1].value == "3"

    assert isinstance(idx_args.value[2], CASTLiteralValue)
    assert idx_args.value[2].value == "2"


def test_list4():
    cast = generate_cast(list4())

    func_def_node = cast.nodes[0].body[0]
    assert isinstance(func_def_node, FunctionDef)
    assert func_def_node.name.name == "foo"
    assert func_def_node.name.id == 0
    assert isinstance(func_def_node.body[0], ModelReturn)
    assert isinstance(func_def_node.body[0].value, CASTLiteralValue)
    assert func_def_node.body[0].value.value == "2"

    asg_node = cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 1

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "List"
    assert len(asg_node.right.value) == 5

    index_node = cast.nodes[0].body[2]
    assert isinstance(index_node, Assignment)
    assert isinstance(index_node.left, Var) 
    assert isinstance(index_node.left.val, Name) 
    assert index_node.left.val.name == "y"
    assert index_node.left.val.id == 3

    index_call = index_node.right
    assert isinstance(index_call, Call)
    assert isinstance(index_call.func, Name)    
    assert index_call.func.name == "_get"
    assert index_call.func.id == 2
    assert len(index_call.arguments) == 2

    slice1 = index_call.arguments[0]
    assert isinstance(slice1, Name)
    assert slice1.name == "x"
    assert slice1.id == 1

    slice2 = index_call.arguments[1]
    assert isinstance(slice2, CASTLiteralValue)
    assert slice2.value_type == "List"
    assert len(slice2.value) == 3 

    assert isinstance(slice2.value[0], CASTLiteralValue)
    assert slice2.value[0].value == "0"

    assert isinstance(slice2.value[1], Call)
    assert slice2.value[1].func.name == "foo"
    assert slice2.value[1].func.id == 0

    assert isinstance(slice2.value[2], CASTLiteralValue)
    assert slice2.value[2].value == "1"
