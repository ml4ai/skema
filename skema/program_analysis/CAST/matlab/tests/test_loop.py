from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    cast_nodes
)

# Test the for loop and others

def no_test_for_loop():
    """ Test the MATLAB for loop syntax elements"""

    source = """
        for n = 1:10
            x = do_something(n)
        end
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
