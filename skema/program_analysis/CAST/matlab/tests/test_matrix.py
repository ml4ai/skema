from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue
)

# Test CAST using matrices
def test_matrix_empty():
    """ Test assignment of empty matrices."""
    nodes = cast("x = [];")
    check(nodes[0], Assignment(left = 'x', right = []))

def test_matrix_boolean():
    """ Test assignment of empty matrices."""
    nodes = cast("x = [true false];")
    check(nodes[0], Assignment(left = 'x', right = ["True", "False"]))

def test_matrix_values():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast("x = [1 x 'Bob' ]")
    assert len(nodes) == 1
    check(nodes[0], Assignment(left = 'x', right = [1, 'x', "'Bob'"]))

