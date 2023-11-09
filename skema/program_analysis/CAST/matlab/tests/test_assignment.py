from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    cast_nodes
)

# Test CAST from assignment

def test_literal():
    """ Test assignment of literal types (number, string, boolean)."""

    source = """
        x = 5
        y = 1.8
        x = 'single'
        y = "double"
        yes = true
        no = false
    """
    
    nodes = cast_nodes(source)
    assert len(nodes) == 6

    # number
    assert_assignment(nodes[0], left = "x", right = "5")
    assert_assignment(nodes[1], left = "y", right = "1.8")
    # string
    assert_assignment(nodes[2], left = "x", right = "'single'")
    assert_assignment(nodes[3], left = "y", right = "\"double\"")
    # boolean
    assert_assignment(nodes[4], left = 'yes', right = 'true')
    assert_assignment(nodes[5], left = 'no', right = 'false')


def test_identifier():
    """ Test assignment of identifiers."""

    source = """
        x = y
    """

    nodes = cast_nodes(source)
    assert len(nodes) == 1

    # identifier
    assert_assignment(nodes[0], left = 'x', right = 'y')
