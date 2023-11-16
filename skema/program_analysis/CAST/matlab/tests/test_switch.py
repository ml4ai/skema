from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import ModelIf

def test_case_clause_1_argument():
    """ Test CAST from single argument case clause."""

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
    assert_expression(mi0.expr, op="==", operands = ["s", "'one'"])
    assert_assignment(mi0.body[0], left="n", right = 1)

    # case clause 'two'
    mi1 = mi0.orelse[0]
    assert isinstance(mi1, ModelIf)
    assert_expression(mi1.expr, op="==", operands = ["s", "'two'"])
    assert_assignment(mi1.body[0], left="n", right = 2)
    assert_assignment(mi1.body[1], left="x", right = "y")

    # otherwise clause
    assert_assignment(mi1.orelse[0], left="n", right = 0)

def test_case_clause_n_arguments():
    """ Test CAST from multipe argument case clause."""

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
    assert_expression(
        mi0.expr,
        op="in",
        operands = ['s', ["'one'", "'two'", "'three'"]]
    )
    assert_assignment(mi0.body[0], left="n", right = 1)

    # case clause 2
    mi1 = mi0.orelse[0]
    assert isinstance(mi1, ModelIf)
    assert_expression(mi1.expr, op = "==", operands = ['s', 2])
    assert_assignment(mi1.body[0], left="n", right = 2)

    # otherwise clause
    assert_assignment(mi1.orelse[0], left="n", right = 0)
