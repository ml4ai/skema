from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_expression,
    first_cast_node
)
from skema.program_analysis.CAST2FN.model.cast import ModelIf

def test_if():
    """ Test CAST from MATLAB 'if' conditional logic."""

    source = """
    if x > 5
        y = 6
    end
    """

    mi = first_cast_node(source)
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">", left = "x", right = "5")
    assert_assignment(mi.body[0], left="y", right = "6")

def test_if_else():
    """  Test CAST from MATLAB 'if else' conditional logic."""

    source = """
    if x > 5
        y = 6
    else
        y = x
    end
    """

    mi = first_cast_node(source)
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">", left = "x", right = "5")
    assert_assignment(mi.body[0], left="y", right = "6")
    # else
    assert_assignment(mi.orelse[0], left="y", right = "x")

def test_if_elseif_else():
    """ Test CAST from MATLAB 'if elseif else' conditional logic."""

    source = """
    if x > 5
        y = 6
    elseif x > 0
        y = x
    else
        y = 0
    end
    """
    
    mi = first_cast_node(source)
    # if
    assert isinstance(mi, ModelIf)
    assert_expression(mi.expr, op = ">", left = "x", right = "5")
    assert_assignment(mi.body[0], left="y", right = "6")
    # elseif
    assert isinstance(mi.orelse[0], ModelIf)
    assert_expression(mi.orelse[0].expr, op = ">", left = "x", right = "0")
    assert_assignment(mi.orelse[0].body[0], left="y", right = "x")
    # else
    assert_assignment(mi.orelse[1], left="y", right = "0")
