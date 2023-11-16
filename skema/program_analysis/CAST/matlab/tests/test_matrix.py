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

def test_matrix_empty_2D():
    """ Test assignment of empty matrices."""
    nodes = cast_nodes("x = [[] []];")
    check_result(nodes[0], Assignment(left = 'x', right = [[], []]))

def test_matrix_empty_3D():
    """ Test assignment of empty matrices."""
    nodes = cast_nodes("x = [[[] []] [[] []]]")
    check_result(nodes[0], Assignment(left = 'x', right = [[[], []], [[], []]]))

def test_matrix_integer():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast_nodes("x = [1 2 3]")
    assert len(nodes) == 1
    check_result(nodes[0], Assignment(left = 'x', right = [1, 2, 3]))

def test_matrix_identifier():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast_nodes("x = [a b c]")
    assert len(nodes) == 1
    check_result(nodes[0], Assignment(left = 'x', right = ['a', 'b', 'c']))

def test_matrix_2D_identifier():
    """ Test assignment 2 dimensional matrix value."""
    nodes = cast_nodes("x = [[a b] [c d]]")
    assert len(nodes) == 1
    check_result(nodes[0], Assignment(left = 'x', right = [['a', 'b'], ['c', 'd']]))

def test_matrix_3D_identifier():
    """ Test assignment 2 dimensional matrix value."""
    nodes = cast_nodes("x = [[[a b] [c d]] [[e f] [g h]]]")
    assert len(nodes) == 1
    check_result(
        nodes[0], 
        Assignment(
            left = 'x', 
            right = [[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]
        )
    )
