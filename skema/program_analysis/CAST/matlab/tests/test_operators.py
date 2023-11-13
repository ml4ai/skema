from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_var,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import Assignment

def test_binary_operator():
    """ Test CAST from MATLAB binary operation statement."""

    source = 'z = x + y'

    # cast nodes should be one assignment
    nodes = cast_nodes(source)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Assignment)

    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")

    # right assignment operand is a binary expression
    assert_expression(nodes[0].right, op = "+", left = "x", right = "y")

def do_not_test_unary_operator():
    """ Test CAST from MATLAB binary operation statement."""

    source = 'z = -6'

    # cast nodes should be one assignment
    nodes = cast_nodes(source)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Assignment)

    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")

    # right assignment operand is a binary expression
    assert_expression(nodes[0].right, op = "+", left = "x", right = "y")
