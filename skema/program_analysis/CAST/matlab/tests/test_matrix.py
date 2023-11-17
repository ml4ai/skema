from skema.program_analysis.CAST.matlab.tests.utils import (
    check_result,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue
)


# Test CAST from assignment

def test_matrix_empty():
    """ Test assignment of empty matrices."""
    nodes = cast_nodes("x = [];")
    check_result(nodes[0], Assignment(left = 'x', right = []))

def test_matrix_boolean():
    """ Test assignment of empty matrices."""
    nodes = cast_nodes("x = [true false];")
    check_result(nodes[0], Assignment(left = 'x', right = ["True", "False"]))

def test_matrix_values():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast_nodes("x = [1 x 'Bob' ]")
    assert len(nodes) == 1
    check_result(nodes[0], Assignment(left = 'x', right = [1, 'x', "'Bob'"]))

