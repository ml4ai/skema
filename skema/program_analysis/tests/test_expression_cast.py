# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    LiteralValue
)

def exp0():
    return """
x = 2
    """

def exp1():
    return """
x = 2
y = 3
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_exp0():
    exp_cast = generate_cast(exp0())
    # Test basic properties of assignment node
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'


def test_exp1():
    exp_cast = generate_cast(exp1())

    # Test basic properties of two assignment nodes
    asg_node = exp_cast.nodes[0].body[0]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'
    
    # ------
    asg_node = exp_cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "y"
    assert asg_node.left.val.id == 1

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'
