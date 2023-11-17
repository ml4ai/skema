from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Operator
)

def operator_only_test(node, operator: Operator):
    """ An assignment of 'x = operator' is presumed """
    check(
        node,
        Assignment(
            left = "x", 
            right = operator
        )
    )

def test_binary_operator():
    """ Test CAST from binary operator."""
    nodes = cast("x = y + z")
    operator_only_test(nodes[0], Operator(op = "+", operands = ["y", "z"]))

def test_boolean_operator():
    """ Test CAST from boolean operator."""
    nodes = cast("x = yes && no")
    operator_only_test(nodes[0], Operator(op = "&&", operands = ["yes", "no"]))

def test_unary_operator():
    """ Test CAST from unary operator."""
    nodes = cast("x = -6;")
    operator_only_test(nodes[0], Operator(op = "-", operands = [6]))

def test_comparison_operator(): 
    """ Test CAST from comparison operator."""
    nodes = cast("x = y < 4")
    operator_only_test(nodes[0], Operator(op = "<", operands = ["y", 4]))

# no test

def no_test_not_operator():
    """ Test CAST from matrix not operator."""
    # logical matrix inversion, implement as nested for loops
    nodes = cast("z = x ~y")
    # CAST should be one Assignment node
    assert_assignment(nodes[0], left = "z")

def no_test_postfix_operator():
    """ Test CAST from postfix operator."""
    # this is a unary operator type
    nodes = cast("z = y'")
    # CAST should be one Assignment node
    assert_assignment(nodes[0], left = "z")
    
def no_test_spread_operator():
    """ Test CAST from matrix not operator."""
    nodes = cast("beta_v1(:,1) = new_beta_v1;")
    # CAST should be one Assignment node

    # right assignment operand an identifier
    assert_identifier(nodes[0].right, "new_beta_v1")
