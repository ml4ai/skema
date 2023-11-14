from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_identifier,
    cast_nodes
)

# Test CAST from assignment

def test_number():
    """ Test assignment of integer and real numbers."""
    nodes = cast_nodes("x = 5; y = 1.8")
    assert len(nodes) == 2
    assert_assignment(nodes[0], left = "x", right = 5)
    assert_assignment(nodes[1], left = "y", right = 1.8)

def test_string():
    """ Test assignment of single and double quoted strings."""
    source = """
    x = 'single'
    y = "double"
    """
    nodes = cast_nodes(source)
    assert len(nodes) == 2
    assert_assignment(nodes[0], left = "x", right = "'single'")
    assert_assignment(nodes[1], left = "y", right = "\"double\"") 

def test_boolean():
    """ Test assignment of literal boolean types. """
    nodes = cast_nodes("x = true; y = false")
    assert len(nodes) == 2
    assert_assignment(nodes[0], left = "x", right = "True")
    assert_assignment(nodes[1], left = "y", right = "False")

def test_identifier():
    """ Test assignment of identifiers."""
    nodes = cast_nodes("x = y; r = x")
    assert len(nodes) == 2
    assert_assignment(nodes[0], left = 'x', right = 'y')
    assert_assignment(nodes[1], left = 'r', right = 'x')

def test_literal_matrix():
    """ Test assignment MATLAB literal matrix value."""
    nodes = cast_nodes("x = [2 3 5]")
    assert len(nodes) == 1
    assert_assignment(nodes[0], left = 'x', right = [2, 3, 5])
