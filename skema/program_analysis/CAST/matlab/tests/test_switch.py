from skema.program_analysis.CAST.matlab.tests.utils import (
    check_result,
    cast_nodes
)
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
    check_result(
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(op = "==", operands = ["s", "'one'"]),
            # then
            body = [
                Assignment(left="n", right = 1)
            ],
            # else
            orelse = ModelIf(
                # if
                expr = Operator(op = "==", operands = ["s", "'two'"]),
                # then
                body = [
                    Assignment(left="n", right = 2),
                    Assignment(left="x", right = "y"),
                ],
                # else
                orelse = [
                    Assignment(left="n", right = 0)
                ]
            )
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
    check_result(
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(
                op = "in",
                operands = ["s", [1, 1.0, "one", "'one'"]]
            ),
            # then
            body = [
                Assignment(left="n", right = 1)
            ],
            # else
            orelse = [
                Assignment(left="n", right = 0)
            ]
        )
    )
