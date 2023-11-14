from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import ModelIf

def test_if():
    """ Test CAST from MATLAB 'if' conditional logic."""

    source = """
    if x == 5
        y = 6
    end
    """

    mi = cast_nodes(source)[0]

    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = "==", operands = ["x", 5])
    assert_assignment(mi.body[0], left="y", right = 6)

def test_if_else():
    """  Test CAST from MATLAB 'if else' conditional logic."""

    source = """
    if x > 5
        y = 6
        three = 3
    else
        y = x
        foo = 'bar'
    end
    """

    mi = cast_nodes(source)[0]
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">", operands = ["x", 5])
    assert_assignment(mi.body[0], left="y", right = 6)
    assert_assignment(mi.body[1], left="three", right = 3)
    # else
    assert_assignment(mi.orelse[0], left="y", right = "x")
    assert_assignment(mi.orelse[1], left="foo", right = "'bar'")

def test_if_elseif():
    """ Test CAST from MATLAB 'if elseif else' conditional logic."""

    source = """
    if x >= 5
        y = 6
    elseif x <= 0
        y = x
    end
    """
    
    mi = cast_nodes(source)[0]
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">=", operands = ["x", 5])
    assert_assignment(mi.body[0], left="y", right = 6)
    # elseif
    assert isinstance(mi.orelse[0], ModelIf)
    assert_expression(mi.orelse[0].expr, op = "<=", operands = ["x", 0])
    assert_assignment(mi.orelse[0].body[0], left="y", right = "x")

def test_if_elseif_else():
    """ Test CAST from MATLAB 'if elseif else' conditional logic."""

    source = """
    if x > 5
        a = 6
    elseif x > 0
        b = x
    else
        c = 0
    end
    """
    
    mi = cast_nodes(source)[0]
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">", operands = ["x", 5])
    assert_assignment(mi.body[0], left="a", right = 6)
    # elseif
    assert isinstance(mi.orelse[0], ModelIf)
    assert_expression(mi.orelse[0].expr, op = ">", operands = ["x", 0])
    assert_assignment(mi.orelse[0].body[0], left="b", right = "x")
    # else
    assert_assignment(mi.orelse[0].orelse[0], left="c", right = 0)
