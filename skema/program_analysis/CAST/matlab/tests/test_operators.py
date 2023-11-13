from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_var,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import Assignment

def test_binary_operator():
    """ Test CAST from binary operator."""
    nodes = cast_nodes("z = x + y")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")
    # right assignment operand represents binary operation
    assert_expression(nodes[0].right, op = "+", operands = ["x", "y"])

def test_boolean_operator():
    """ Test CAST from boolean operator."""
    nodes = cast_nodes("z = x && y")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")
    # right assignment operand represents binary operation
    assert_expression(nodes[0].right, op = "&&", operands = ["x", "y"])

def test_comparison_operator(): 
    """ Test CAST from comparison operator."""
    nodes = cast_nodes("z = x < y")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "z")
    # right assignment operand represents comparison operation
    assert_expression(nodes[0].right, op = "<", operands = ["x", "y"])

def no_test_not_operator():
    """ Test CAST from matrix not operator."""
    # logical matrix inversion, implement as nested for loops
    nodes = cast_nodes("z = x ~y")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)

def no_test_postfix_operator():
    """ Test CAST from postfix operator."""
    """"
    SOURCE (op.m):
    x = y'

    SYNTAX TREE:
    ERROR
      assignment
        identifier
        =
        postfix_operator
          identifier
          '
    """
    # this is a unary operator type

    nodes = cast_nodes("z = y'")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    
def no_test_spread_operator():
    """ Test CAST from matrix not operator."""

    """
    SOURCE:
    beta_v1(:,1) = new_beta_v1;

    SYNTAX TREE:
    assignment
        function_call
            identifier
            (
            arguments
                spread_operator
                    :
                ,
                number
            )
        =
        identifier
    ;
    """

    nodes = cast_nodes("beta_v1(:,1) = new_beta_v1;")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand a function call
    # assert_var(nodes[0].left, name = "a")

    # right assignment operand an identifier
    assert_var(nodes[0].right, "new_beta_v1")

def test_unary_operator():
    """ Test CAST from unary operator."""
    nodes = cast_nodes("x = -6;")
    # CAST should be one Assignment node
    assert isinstance(nodes[0], Assignment)
    # Left assignment operand is the variable
    assert_var(nodes[0].left, name = "x")
    # right assignment operand is a unary operator
    assert_expression(nodes[0].right, op = "-", operands = ["6"])

