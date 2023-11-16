from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_loop,
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
    assert len (nodes) == 1
    assert_loop(nodes[0])
