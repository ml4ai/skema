from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    cast_nodes
)

# Test the CAST returned by processing the simplest MATLAB assignment

def test_assignment():
    """ Test CAST from MATLAB 'assignment' statement."""

    source = """
    x = 5
    y = "xxx"
    """
    
    # nodes should be two assignments
    nodes = cast_nodes(source)
    assert len(nodes) == 2
    assert_assignment(nodes[0], left = "x", right = "5")

    # When comparing strings you must include escaped quotes
    assert_assignment(nodes[1], left = "y", right = "\"xxx\"")
