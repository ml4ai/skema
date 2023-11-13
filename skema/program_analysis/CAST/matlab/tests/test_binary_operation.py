from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_var,
    assert_expression,
    first_cast_node
)
from skema.program_analysis.CAST2FN.model.cast import Assignment

# Test the CAST returned by processing the simplest MATLAB binary operation

def test_binary_operation():
    """ Test CAST from MATLAB binary operation statement."""

    source = 'z = x + y'

    # The root of the CAST should be Assignment
    assignment = first_cast_node(source)
    assert isinstance(assignment, Assignment)

    # Left operand of this assignment node is the variable
    assert_var(assignment.left, name = "z")

    # right operand of this assignment node is a binary expression
    assert_expression(assignment.right, op = "+", left = "x", right = "y")
