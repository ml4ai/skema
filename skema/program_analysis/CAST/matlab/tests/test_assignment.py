from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Operator
)

# Test CAST from assignment

def test_boolean():
    """ Test assignment of literal boolean types. """
    # we translate these MATLAB keywords into capitalized strings for Python
    nodes = cast("x = true; y = false")
    check(nodes[0], Assignment(left = "x", right = "True"))
    check(nodes[1], Assignment(left = "y", right = "False"))

def test_number_zero_integer():
    """ Test assignment of integer and real numbers."""
    check(cast("x = 0")[0], Assignment(left = "x", right = 0))

def test_number_zero_real():
    """ Test assignment of integer and real numbers."""
    check(cast("y = 0.0")[0], Assignment(left = "y", right = 0.0))

def test_number_nonzero():
    """ Test assignment of integer and real numbers."""
    check(cast("z = 1.8")[0], Assignment(left = "z", right = 1.8))

def test_string():
    """ Test assignment of single and double quoted strings."""
    source = """
    x = 'single'
    y = "double"
    """
    nodes = cast(source)
    check(nodes[0], Assignment(left = "x", right = "'single'"))
    check(nodes[1], Assignment(left = "y", right = "\"double\""))

def test_identifier():
    """ Test assignment of identifiers."""
    nodes = cast("x = y; r = x")
    check(nodes[0], Assignment(left = 'x', right = 'y'))
    check(nodes[1], Assignment(left = 'r', right = 'x'))

def test_operator():
    """ Test assignment of operator"""
    check(
        cast("Vtot = V1PF+V1AZ;")[0], 
        Assignment(
            left = "Vtot", 
            right = Operator(
                op = "+",
                operands = ["V1PF", "V1AZ"]
            )
        )
    )
