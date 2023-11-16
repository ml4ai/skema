from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_call,
    assert_operand,
    cast_nodes
)

def test_command():
    """ Test the MATLAB command syntax elements"""

    nodes = cast_nodes("clear all;")
    assert len (nodes) == 1

    assert_call(nodes[0], func = "clear")
    assert_operand(nodes[0].arguments[0], "all")
