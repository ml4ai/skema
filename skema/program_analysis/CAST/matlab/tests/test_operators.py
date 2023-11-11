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
    assert_expression(nodes[0].right, op = "+", left = "x", right = "y")

def test_not_operator():
    """ Test CAST from matrix not operator."""

    source = 'a = ~mat_val'

    nodes = cast_nodes(source)
    assert len(nodes) == 1

    # The expression becomes a CAST Assignment
    assert_assignment(nodes[0], left = "a", right = "~mat_val")

def test_unary_operator_literal():
    """ Test CAST from unary operator."""

    source = 'x = -6'

    nodes = cast_nodes(source)
    assert len(nodes) == 1

    # Test with literal
    assert_assignment(nodes[0], left = "x", right = "-6")

def test_unary_operator_identifier():
    """ Test CAST from unary operator."""

    source = 'y = -x'

    nodes = cast_nodes(source)
    assert len(nodes) == 1

    # Test with identifier
    assert_assignment(nodes[0], left = "y", right = "-x")

def test_unary_operator():
    """ Test CAST from unary operator."""

    source = """
        x = -6
        y = -x
        end
    """

    nodes = cast_nodes(source)
    # assert len(nodes) == 2

    # Test with literal and identifier
    assert_assignment(nodes[0], left = 'x', right = '-6')
    assert_assignment(nodes[1], left = 'y', right = '-x')
