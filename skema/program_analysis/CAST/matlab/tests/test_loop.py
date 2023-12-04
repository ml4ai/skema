from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    Loop,
    Operator
)

# Test the for loop incrementing by 1
def test_implicit_step():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 0:10
            x = disp(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [Assignment(left = "n", right = 0)],
            expr = Operator(op = "<=", operands = ["n", 10]),
            body = [
                Assignment(
                    left = "n",
                    right = Operator(
                        op = "+",
                        operands = ["n", 1]
                    )
                ),
                Assignment(
                    left = "x",
                    right = Call(
                        func = "disp",
                        arguments = ["n"]
                    )
                )
            ],
            post = []
        )
    )

# Test the for loop incrementing by n
def test_explicit_step():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 0:2:10
            x = disp(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [Assignment(left = "n", right = 0)],
            expr = Operator(op = "<=", operands = ["n", 10]),
            body = [
                Assignment(
                    left = "n",
                    right = Operator(
                        op = "+",
                        operands = ["n", 2]
                    )
                ),
                Assignment(
                    left = "x",
                    right = Call(
                        func = "disp",
                        arguments = ["n"]
                    )
                )
            ],
            post = []
        )
    )
