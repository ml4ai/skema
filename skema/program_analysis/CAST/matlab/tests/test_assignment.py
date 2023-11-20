from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import Assignment

# Test CAST from assignment

def test_boolean():
    """ Test assignment of literal boolean types. """
    nodes = cast("x = true; y = false")
    check(nodes[0], Assignment(left = "x", right = "True"))
    check(nodes[1], Assignment(left = "y", right = "False"))

def test_number_zero_integer():
    """ Test assignment of integer and real numbers."""
    nodes = cast("x = 0")
    check(nodes[0], Assignment(left = "x", right = 0))

def test_number_zero_real():
    """ Test assignment of integer and real numbers."""
    nodes = cast("y = 0.0")
    check(nodes[0], Assignment(left = "y", right = 0.0))

def test_number_nonzero():
    """ Test assignment of integer and real numbers."""
    nodes = cast("z = 1.8")
    check(nodes[0], Assignment(left = "z", right = 1.8))

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
