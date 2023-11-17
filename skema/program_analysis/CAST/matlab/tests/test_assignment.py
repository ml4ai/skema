from skema.program_analysis.CAST.matlab.tests.utils import (
    check_result,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import Assignment


# Test CAST from assignment

def test_boolean():
    """ Test assignment of literal boolean types. """
    nodes = cast_nodes("x = true; y = false")
    check_result(nodes[0], Assignment(left = "x", right = "True"))
    check_result(nodes[1], Assignment(left = "y", right = "False"))

def test_number_zero_integer():
    """ Test assignment of integer and real numbers."""
    nodes = cast_nodes("x = 0")
    check_result(nodes[0], Assignment(left = "x", right = 0))

def test_number_zero_real():
    """ Test assignment of integer and real numbers."""
    nodes = cast_nodes("y = 0.0")
    check_result(nodes[0], Assignment(left = "y", right = 0.0))

def test_number_nonzero():
    """ Test assignment of integer and real numbers."""
    nodes = cast_nodes("z = 1.8")
    check_result(nodes[0], Assignment(left = "z", right = 1.8))

def test_string():
    """ Test assignment of single and double quoted strings."""
    source = """
    x = 'single'
    y = "double"
    """
    nodes = cast_nodes(source)
    check_result(nodes[0], Assignment(left = "x", right = "'single'"))
    check_result(nodes[1], Assignment(left = "y", right = "\"double\""))

def test_identifier():
    """ Test assignment of identifiers."""
    nodes = cast_nodes("x = y; r = x")
    check_result(nodes[0], Assignment(left = 'x', right = 'y'))
    check_result(nodes[1], Assignment(left = 'r', right = 'x'))
