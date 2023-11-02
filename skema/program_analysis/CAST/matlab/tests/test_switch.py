from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_var,
    assert_expression,
    cast_nodes
)
from skema.program_analysis.CAST2FN.model.cast import Assignment

def do_not_test_switch():
    """ Test CAST from MATLAB switch statement."""

    source = """
    switch s
        case 'axis'
            n = 0;
        case 'bottom'
            n = 1;
        case 'top'
            n = 2;
    end
    """

    # The root of the CAST should be Assignment
    nodes = cast_nodes(source)
