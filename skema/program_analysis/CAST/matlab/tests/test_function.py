from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_call,
    assert_identifier,
    cast_nodes
)

# Test CAST from functions

def test_function_definition():
    """ Test function """

    source = """
    function both = add_them(x, y)
        both = x + y
    end
    """

    nodes = cast_nodes(source)
    assert len(nodes) == 1


def test_function_call():
    """ Test function call """

    nodes = cast_nodes("z = both(3, 5)")
    assert len(nodes) == 1
    assert_assignment(nodes[0], left='z')
    assert_call(nodes[0].right, func = "both", arguments = [3, 5])
