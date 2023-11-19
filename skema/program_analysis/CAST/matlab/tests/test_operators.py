from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    Operator
)

def test_binary_operator():
    """ Test CAST from binary operator."""
    check(cast("y + z;")[0], Operator(op = "+", operands = ["y","z"]))

def test_boolean_operator():
    """ Test CAST from boolean operator."""
    check(cast("yes && no;")[0], Operator(op = "&&", operands = ["yes","no"]))

def test_unary_operator():
    """ Test CAST from unary operator."""
    check(cast("-6;")[0], Operator(op = "-", operands = [6]))

def test_comparison_operator(): 
    """ Test CAST from comparison operator."""
    check(cast("y < 4;")[0], Operator(op = "<", operands = ["y", 4]))

def test_not_operator():
    """ Test CAST from matrix not operator."""
    check(cast("~y")[0], Operator(op = "~", operands = ["y"]))

def test_postfix_operator():
    """ Test CAST from postfix operator."""
    check(cast("y'")[0], Operator(op = "'", operands = ["y"]))
    
def test_spread_operator():
    """ Test CAST from spread operator."""
    check(
        cast("foo(:)")[0], 
        Call(
            func = "foo",
            arguments = [
                Operator(
                    op = ":",
                    operands = []
                )
            ]
        )
    )
