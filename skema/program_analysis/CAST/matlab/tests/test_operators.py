from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_var,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import Assignment

def test_binary_operator():
    """ Test CAST from binary operator."""

    source = 'z = x + y'

    # cast nodes should be one assignment
    nodes = cast_nodes(source)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Assignment)

    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")
    # right assignment operand is a binary expression
    assert_expression(nodes[0].right, op = "+", operands = ["x", "y"])

def test_not_operator():
    """ Test CAST from matrix not operator."""

    source = 'a = ~mat_val'

    # we should have one Assignment
    nodes = cast_nodes(source)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Assignment)

    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "a")
    # right assignment operand is a unary operator
    assert_expression(nodes[0].right, op = "~", operands = ["mat_val"])

def test_unary_operator():
    """ Test CAST from unary operator."""

    source = """
        x = -6;
        y = -x;
    """

    nodes = cast_nodes(source)
    assert len(nodes) == 2

    # Line 1
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "x")
    # right assignment operand is a unary operator
    assert_expression(nodes[0].right, op = "-", operands = ["6"])

    # Line 2
    assert isinstance(nodes[1], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[1].left, name = "y")
    # right assignment operand is a unary operator
    assert_expression(nodes[1].right, op = "-", operands = ["x"])

