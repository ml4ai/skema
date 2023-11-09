from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    cast_nodes
)

# Test assignment of different datatypes

def test_numeric():
    """ Test assignment of integer and real numbers."""

    source = """
        x = 5;
        y = 1.8;
    """
    
    nodes = cast_nodes(source)
    assert len(nodes) == 2

    # integer
    assert_assignment(nodes[0], left = "x", right = "5")
    # real
    assert_assignment(nodes[1], left = "y", right = "1.8")


def test_string():
    """ Test assignment of strings. """


    source = """
        x = 'cat';
        y = "dog";
    """

    nodes = cast_nodes(source)
    assert len(nodes) == 2

    # single quotes
    assert_assignment(nodes[0], left = "x", right = "'cat'")
    # double quotes
    assert_assignment(nodes[1], left = "y", right = "\"dog\"")
