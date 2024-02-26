# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Call,
    Name,
    CASTLiteralValue,
    ModelIf,
    Loop,
    Operator
)

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
    x = x + 1
    x = x + y
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_while1():
    cast = generate_cast(while1())

    asg_node = cast.nodes[0].body[0]
    loop_node = cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'

    assert isinstance(loop_node, Loop)
    assert len(loop_node.pre) == 0

    # Loop Test
    loop_test = loop_node.expr 
    assert isinstance(loop_test, Operator)
    assert loop_test.op == "ast.Lt"
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "x"

    assert isinstance(loop_test.operands[1], CASTLiteralValue)
    assert loop_test.operands[1].value_type == "Integer"
    assert loop_test.operands[1].value == "5"

    # Loop Body
    loop_body = loop_node.body
    asg = loop_body[0]
    assert isinstance(asg, Assignment)
    assert isinstance(asg.left, Var)
    assert asg.left.val.name == "x"

    assert isinstance(asg.right, Operator)
    assert asg.right.op == "ast.Add"
    assert isinstance(asg.right.operands[0], Name)
    assert isinstance(asg.right.operands[1], CASTLiteralValue)
    assert asg.right.operands[1].value == "1"

def test_while2():
    cast = generate_cast(while2())

    asg_node = cast.nodes[0].body[0]
    asg_node_2 = cast.nodes[0].body[1]
    loop_node = cast.nodes[0].body[2]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'

    assert isinstance(asg_node_2, Assignment)
    assert isinstance(asg_node_2.left, Var)
    assert isinstance(asg_node_2.left.val, Name)
    assert asg_node_2.left.val.name == "y"

    assert isinstance(asg_node_2.right, CASTLiteralValue)
    assert asg_node_2.right.value_type == "Integer"
    assert asg_node_2.right.value == '3'

    assert isinstance(loop_node, Loop)
    assert len(loop_node.pre) == 0

    # Loop Test
    loop_test = loop_node.expr 
    assert isinstance(loop_test, Operator)
    assert loop_test.op == "ast.Lt"
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "x"

    assert isinstance(loop_test.operands[1], CASTLiteralValue)
    assert loop_test.operands[1].value_type == "Integer"
    assert loop_test.operands[1].value == "5"

    # Loop Body
    loop_body = loop_node.body
    asg = loop_body[0]
    assert isinstance(asg, Assignment)
    assert isinstance(asg.left, Var)
    assert asg.left.val.name == "x"

    assert isinstance(asg.right, Operator)
    assert asg.right.op == "ast.Add"
    assert isinstance(asg.right.operands[0], Name)
    assert asg.right.operands[0].name == "x"

    assert isinstance(asg.right.operands[1], CASTLiteralValue)
    assert asg.right.operands[1].value == "1"

    asg = loop_body[1]
    assert isinstance(asg, Assignment)
    assert isinstance(asg.left, Var)
    assert asg.left.val.name == "x"

    assert isinstance(asg.right, Operator)
    assert asg.right.op == "ast.Add"
    assert isinstance(asg.right.operands[0], Name)
    assert asg.right.operands[0].name == "x"

    assert isinstance(asg.right.operands[1], Name)
    assert asg.right.operands[1].name == "y"

