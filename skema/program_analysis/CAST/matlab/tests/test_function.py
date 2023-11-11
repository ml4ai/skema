from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    cast_nodes
)

# Test CAST from functions

def no_test_function():
    """ Test function """

    source = """
        function both = add_them(x, y)
            both = x + y
    end
         = false
    """
    

    nodes = cast_nodes(source)
    assert len(nodes) == 2

    # identifier
    assert_assignment(nodes[0], left = 'x', right = 'y')
    assert_assignment(nodes[1], left = 'r', right = 'x')
