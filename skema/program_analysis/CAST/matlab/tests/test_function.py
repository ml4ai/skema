from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_foo,
    cast_nodes
)

from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call
)


# Test CAST from functions

def no_test_function_definition():
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
    assert_foo(
        nodes[0],
        Assignment(
            left = "z",
            right = Call(
                func = "both", 
                arguments = [3, 5]
            )
        )
    )

