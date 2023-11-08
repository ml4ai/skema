from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import ModelIf

def test_switch_single_values():
    """ Test CAST from MATLAB switch statement."""

    source = """
    switch s
        case 'one'
            n = 1;
        case 'two'
            n = 2;
            x = y;
        otherwise
            n = 0;
    end
    """

    # case clause 'one'
    mi0 = cast_nodes(source)[0]
    assert isinstance(mi0, ModelIf)
    assert_assignment(mi0.body[0], left="n", right = "1")
    assert_expression(mi0.expr, op="==", left = "s", right = "'one'")

    # case clause 'two'
    mi1 = mi0.orelse[0]
    assert isinstance(mi1, ModelIf)
    assert_assignment(mi1.body[0], left="n", right = "2")
    assert_assignment(mi1.body[1], left="x", right = "y")

    # otherwise clause
    assert_assignment(mi1.orelse[0], left="n", right = "0")

def test_switch_multiple_values():
    """ Test CAST from MATLAB switch statement."""

    source = """
    switch s
        case {'one', 'two', 'three'}
            n = 1;
        case 2
            n = 2;
        otherwise
            n = 0;
    end
    """

    # case clause {'one', 'two', 'three'}
    mi0 = cast_nodes(source)[0]
    assert isinstance(mi0, ModelIf)
    assert_assignment(mi0.body[0], left="n", right = "1")
    assert_expression(
        mi0.expr,
        op="in",
        left = 's',
        right = ["'one'", "'two'", "'three'"]
    )

    # case clause 2
    mi1 = mi0.orelse[0]
    assert isinstance(mi1, ModelIf)
    assert_assignment(mi1.body[0], left="n", right = "2")
    assert_expression(mi1.expr, op="==", left = 's', right = 2)

    # otherwise clause
    assert_assignment(mi1.orelse[0], left="n", right = "0")
