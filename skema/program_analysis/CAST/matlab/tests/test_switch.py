from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    ModelIf,
    Operator
)

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
    # switch statement translated into conditional
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(op = "==", operands = ["s", "'one'"]),
            body = [Assignment(left="n", right = 1)],
            orelse = [
                ModelIf(
                    expr = Operator(op = "==", operands = ["s", "'two'"]),
                    body = [
                        Assignment(left="n", right = 2),
                        Assignment(left="x", right = "y"),
                    ],
                    orelse = [Assignment(left="n", right = 0)]
                )
            ]
        )
    )
                       
def test_case_clause_n_arguments():
    """ Test CAST from multipe argument case clause."""

    source = """
    switch s
        case {1, 1.0, one, 'one'}
            n = 1;
        otherwise
            n = 0;
    end
    """
    # switch statement translated into conditional
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(
                op = "in",
                operands = ["s", [1, 1.0, "one", "'one'"]]
            ),
            body = [Assignment(left="n", right = 1)],
            orelse = [Assignment(left="n", right = 0)]
        )
    )
