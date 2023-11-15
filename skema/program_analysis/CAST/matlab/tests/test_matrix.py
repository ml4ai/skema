from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_identifier,
    cast_nodes
)

# Test CAST from assignment

def test_matrix_empty():
    """ Test assignment of empty matrices."""
    nodes = cast_nodes("x = []; y = [[] []]; z = [[[] []] [[] []]]")
    assert len(nodes) == 3
    assert_assignment(nodes[0], left = 'x', right = [])
    assert_assignment(nodes[1], left = 'y', right = [])
    assert_assignment(nodes[2], left = 'z', right = [])

def test_matrix_integer():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast_nodes("x = [1 2 3]")
    assert len(nodes) == 1
    assert_assignment(nodes[0], left = 'x', right = [1, 2, 3])

def test_matrix_identifier():
    """ Test assignment 1 dimensional matrix value."""
    nodes = cast_nodes("x = [a b c]")
    assert len(nodes) == 1
    assert_assignment(nodes[0], left = 'x', right = ['a', 'b', 'c'])

def test_matrix_2D_identifier():
    """ Test assignment 2 dimensional matrix value."""
    nodes = cast_nodes("x = [[a b] [c d]]")
    assert len(nodes) == 1
    assert_assignment(nodes[0], left = 'x', right = [['a', 'b'], ['c', 'd']])

def test_matrix_3D_identifier():
    """ Test assignment 2 dimensional matrix value."""
    nodes = cast_nodes("x = [[[a b] [c d]] [[e f] [g h]]]")
    assert len(nodes) == 1
    assert_assignment(
        nodes[0], 
        left = 'x', 
        right = [[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]
    )
