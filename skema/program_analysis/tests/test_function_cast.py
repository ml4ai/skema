# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    FunctionDef,
    Call,
    LiteralValue,
    Operator,
    ModelReturn
)

def fun1():
    return """
def foo(x):
    return x + 2
x = foo(2)
    """

def fun2():
    return """
def foo(x):
    return x + 3
    
x = foo(2)
y = foo(x)
    """

def fun3():
    return """
def foo(x):
    return x + 3
    
x = foo(2)
y = foo(x)
    """

def fun4():
    return """
def foo(x, a):
    y = x + 3
    z = a * y
    return z
    
x = foo(2, 1)
    """


def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_fun1():
    fun_cast = generate_cast(fun1())
    func_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(func_def_node, FunctionDef)

    assert len(func_def_node.func_args) == 1
    assert isinstance(func_def_node.func_args[0], Var)
    assert func_def_node.func_args[0].val.name == "x"
    assert func_def_node.func_args[0].val.id == 1

    func_def_body = func_def_node.body[0]
    assert isinstance(func_def_body, ModelReturn)
    assert isinstance(func_def_body.value, Operator)
    assert isinstance(func_def_body.value.operands[0], Name)

    assert func_def_body.value.op == "ast.Add"
    assert func_def_body.value.operands[0].name == "x"
    assert func_def_body.value.operands[0].id == 1

    assert isinstance(func_def_body.value.operands[1], LiteralValue)
    assert func_def_body.value.operands[1].value == "2"

    #######################################################

    func_asg_node = fun_cast.nodes[0].body[1]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "x"
    assert func_asg_node.left.val.id == 2

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 1
    assert isinstance(func_call_node.arguments[0], LiteralValue)
    assert func_call_node.arguments[0].value_type == "Integer"
    assert func_call_node.arguments[0].value == "2"


def test_fun2():
    fun_cast = generate_cast(fun2())

    func_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(func_def_node, FunctionDef)

    assert len(func_def_node.func_args) == 1
    assert isinstance(func_def_node.func_args[0], Var)
    assert func_def_node.func_args[0].val.name == "x"
    assert func_def_node.func_args[0].val.id == 1

    func_def_body = func_def_node.body[0]
    assert isinstance(func_def_body, ModelReturn)
    assert isinstance(func_def_body.value, Operator)
    assert isinstance(func_def_body.value.operands[0], Name)

    assert func_def_body.value.op == "ast.Add"
    assert func_def_body.value.operands[0].name == "x"
    assert func_def_body.value.operands[0].id == 1

    assert isinstance(func_def_body.value.operands[1], LiteralValue)
    assert func_def_body.value.operands[1].value == "3"

    #######################################################

    func_asg_node = fun_cast.nodes[0].body[1]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "x"
    assert func_asg_node.left.val.id == 2

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 1
    assert isinstance(func_call_node.arguments[0], LiteralValue)
    assert func_call_node.arguments[0].value_type == "Integer"
    assert func_call_node.arguments[0].value == "2"

    #######################################################

    func_asg_node = fun_cast.nodes[0].body[2]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "y"
    assert func_asg_node.left.val.id == 3

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 1
    assert isinstance(func_call_node.arguments[0], Name)
    assert func_call_node.arguments[0].name == "x"
    assert func_call_node.arguments[0].id == 2


def test_fun3():
    fun_cast = generate_cast(fun3())

    func_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(func_def_node, FunctionDef)

    assert len(func_def_node.func_args) == 1
    assert isinstance(func_def_node.func_args[0], Var)
    assert func_def_node.func_args[0].val.name == "x"
    assert func_def_node.func_args[0].val.id == 1

    func_def_body = func_def_node.body[0]
    assert isinstance(func_def_body, ModelReturn)
    assert isinstance(func_def_body.value, Operator)
    assert isinstance(func_def_body.value.operands[0], Name)

    assert func_def_body.value.op == "ast.Add"
    assert func_def_body.value.operands[0].name == "x"
    assert func_def_body.value.operands[0].id == 1

    assert isinstance(func_def_body.value.operands[1], LiteralValue)
    assert func_def_body.value.operands[1].value == "3"

    #######################################################

    func_asg_node = fun_cast.nodes[0].body[1]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "x"
    assert func_asg_node.left.val.id == 2

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 1
    assert isinstance(func_call_node.arguments[0], LiteralValue)
    assert func_call_node.arguments[0].value_type == "Integer"
    assert func_call_node.arguments[0].value == "2"

    #######################################################

    func_asg_node = fun_cast.nodes[0].body[2]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "y"
    assert func_asg_node.left.val.id == 3

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 1
    assert isinstance(func_call_node.arguments[0], Name)
    assert func_call_node.arguments[0].name == "x"
    assert func_call_node.arguments[0].id == 2

def test_fun4():
    fun_cast = generate_cast(fun4())

    func_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(func_def_node, FunctionDef)

    assert len(func_def_node.func_args) == 2
    assert isinstance(func_def_node.func_args[0], Var)
    assert func_def_node.func_args[0].val.name == "x"
    assert func_def_node.func_args[0].val.id == 1

    assert isinstance(func_def_node.func_args[1], Var)
    assert func_def_node.func_args[1].val.name == "a"
    assert func_def_node.func_args[1].val.id == 2

    func_def_body = func_def_node.body[0]
    assert isinstance(func_def_body, Assignment)
    assert isinstance(func_def_body.left, Var)
    assert func_def_body.left.val.name == "y"
    assert func_def_body.left.val.id == 3

    assert isinstance(func_def_body.right, Operator)
    assert func_def_body.right.op == "ast.Add"

    assert isinstance(func_def_body.right.operands[0], Name)
    assert func_def_body.right.operands[0].name == "x"
    assert func_def_body.right.operands[0].id == 1

    assert isinstance(func_def_body.right.operands[1], LiteralValue)
    assert func_def_body.right.operands[1].value == "3"
    assert func_def_body.right.operands[1].value_type == "Integer"

    
    func_def_body = func_def_node.body[1]
    assert isinstance(func_def_body, Assignment)
    assert isinstance(func_def_body.left, Var)
    assert func_def_body.left.val.name == "z"
    assert func_def_body.left.val.id == 4

    assert isinstance(func_def_body.right, Operator)
    assert func_def_body.right.op == "ast.Mult"

    assert isinstance(func_def_body.right.operands[0], Name)
    assert func_def_body.right.operands[0].name == "a"
    assert func_def_body.right.operands[0].id == 2

    assert isinstance(func_def_body.right.operands[1], Name)
    assert func_def_body.right.operands[1].name == "y"
    assert func_def_body.right.operands[1].id == 3

    func_def_body = func_def_node.body[2]
    assert isinstance(func_def_body, ModelReturn)
    assert isinstance(func_def_body.value, Var)

    assert func_def_body.value.val.name == "z"
    assert func_def_body.value.val.id == 4

    #######################################################
    func_asg_node = fun_cast.nodes[0].body[1]
    assert isinstance(func_asg_node, Assignment)
    assert isinstance(func_asg_node.left, Var)
    assert func_asg_node.left.val.name == "x"
    assert func_asg_node.left.val.id == 5

    func_call_node = func_asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "foo"
    assert func_call_node.func.id == 0

    assert len(func_call_node.arguments) == 2
    assert isinstance(func_call_node.arguments[0], LiteralValue)
    assert func_call_node.arguments[0].value_type == "Integer"
    assert func_call_node.arguments[0].value == "2"

    assert isinstance(func_call_node.arguments[1], LiteralValue)
    assert func_call_node.arguments[1].value_type == "Integer"
    assert func_call_node.arguments[1].value == "1"


