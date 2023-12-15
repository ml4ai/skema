# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    LiteralValue,
    ModelIf,
    Operator
)

def cond1():
    return """
x = 2

if x < 5:
    x = x + 1
else:
    x = x - 3
    """

def cond2():
    return """
x = 2
y = 3

if x < 5:
    x = 1
    y = 2
    x = x * y
else:
    x = x - 3    
    """

def cond3():
    return """
x = 2
y = 4

if x < 5:
    x = x + y
    y = 1
elif x > 10:
    y = x + 2
    x = 1
elif x == 30:
    x = 1
    y = 2
    z = x * y
else:
    x = 0
    y = x - 2
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_cond1():
    exp_cast = generate_cast(cond1())
    
    # Test basic conditional
    asg_node = exp_cast.nodes[0].body[0]
    cond_node = exp_cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'

    assert isinstance(cond_node, ModelIf)
    cond_expr = cond_node.expr
    cond_body = cond_node.body
    cond_else = cond_node.orelse

    assert isinstance(cond_expr, Operator)
    assert cond_expr.op == "ast.Lt"
    assert isinstance(cond_expr.operands[0], Name)
    assert isinstance(cond_expr.operands[1], LiteralValue)

    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert isinstance(cond_body[0].right, Operator)
    assert cond_body[0].right.op == "ast.Add" 

    assert len(cond_else) == 1
    assert isinstance(cond_else[0], Assignment)
    assert isinstance(cond_else[0].left, Var)
    assert isinstance(cond_else[0].right, Operator)
    assert cond_else[0].right.op == "ast.Sub" 


def test_cond2():
    exp_cast = generate_cast(cond2())

    # Test multiple variable conditional
    asg_node = exp_cast.nodes[0].body[0]
    cond_node = exp_cast.nodes[0].body[2]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'
    
    asg_node = exp_cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "y"
    assert asg_node.left.val.id == 1

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'

    assert isinstance(cond_node, ModelIf)
    cond_expr = cond_node.expr
    cond_body = cond_node.body
    cond_else = cond_node.orelse

    assert isinstance(cond_expr, Operator)
    assert cond_expr.op == "ast.Lt"
    assert isinstance(cond_expr.operands[0], Name)
    assert cond_expr.operands[0].name == "x"
    assert isinstance(cond_expr.operands[1], LiteralValue)
    assert cond_expr.operands[1].value_type == "Integer"
    assert cond_expr.operands[1].value == "5"

    assert len(cond_body) == 3
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "x"
    assert isinstance(cond_body[0].right, LiteralValue)
    assert cond_body[0].right.value == "1"

    assert isinstance(cond_body[1], Assignment)
    assert isinstance(cond_body[1].left, Var)
    assert cond_body[1].left.val.name == "y"
    assert isinstance(cond_body[1].right, LiteralValue)
    assert cond_body[1].right.value == "2"

    assert isinstance(cond_body[2], Assignment)
    assert isinstance(cond_body[2].left, Var)
    assert isinstance(cond_body[2].right, Operator)

    assert cond_body[2].right.op == "ast.Mult" 

    assert isinstance(cond_body[2].right.operands[0], Name)
    assert cond_body[2].right.operands[0].name == "x"
    assert cond_body[2].right.operands[0].id == 0
    assert isinstance(cond_body[2].right.operands[1], Name)
    assert cond_body[2].right.operands[1].name == "y"
    assert cond_body[2].right.operands[1].id == 1

    assert len(cond_else) == 1
    assert isinstance(cond_else[0], Assignment)
    assert isinstance(cond_else[0].left, Var)
    assert isinstance(cond_else[0].right, Operator)
    assert cond_else[0].right.op == "ast.Sub" 

def test_cond3():
    exp_cast = generate_cast(cond3())

    # Test nested ifs
    cond_node = exp_cast.nodes[0].body[2]

    assert isinstance(cond_node, ModelIf)
    cond_body = cond_node.body
    cond_else = cond_node.orelse

    assert len(cond_body) == 2
    assert len(cond_else) == 1
    assert isinstance(cond_else[0], ModelIf)
    nested_if = cond_else[0]
    cond_body = nested_if.body
    cond_else = nested_if.orelse

    assert len(cond_body) == 2
    assert len(cond_else) == 1
    assert isinstance(cond_else[0], ModelIf)
    nested_if = cond_else[0]
    cond_body = nested_if.body
    cond_else = nested_if.orelse

    assert len(cond_body) == 3
    assert len(cond_else) == 2
    assert isinstance(cond_else[0], Assignment)
    assert isinstance(cond_else[1], Assignment)
