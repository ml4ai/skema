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
            x = y
        otherwise
            n = 0;
    end
    """

    # The root of the CAST should a ModelIf instance
    mi0 = cast_nodes(source)[0]

    # case 'one'
    assert isinstance(mi0, ModelIf)
    assert_assignment(mi0.body[0], left="n", right = "1")
    assert_expression(mi0.expr, op="==", left = "s", right = "'one'")

    # case 'two'
    mi1 = mi0.orelse[0]
    assert isinstance(mi1, ModelIf)
    assert_assignment(mi1.body[0], left="n", right = "2")
    assert_assignment(mi1.body[1], left="x", right = "y")

    # otherwise
    assert_assignment(mi1.orelse[0], left="n", right = "0")
