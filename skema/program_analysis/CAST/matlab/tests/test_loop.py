from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    Loop,
    Operator
)

# Test the for loop incrementing by 1
def no_test_implicit_step():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 0:10
            disp(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [Assignment(left = "n", right = 0)],
            expr = Operator(op = "<=", operands = ["n", 10]),
            body = [
                Call(
                    func = "disp",
                    arguments = ["n"]
                ),
                Assignment(
                    left = "n",
                    right = Operator(
                        op = "+",
                        operands = ["n", 1]
                    )
                )
            ],
            post = []
        )
    )

# Test the for loop incrementing by n
def no_test_explicit_step():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 0:2:10 
            disp(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [Assignment(left = "n", right = 0)],
            expr = Operator(op = "<=", operands = ["n", 10]),
            body = [
                Call(
                    func = "disp",
                    arguments = ["n"]
                ),
                Assignment(
                    left = "n",
                    right = Operator(
                        op = "+",
                        operands = ["n", 2]
                    )
                )
            ],
            post = []
        )
    )




# Test the for loop using matrix steps
def no_test_matrix():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for k = [10 3 5 6]
            disp(k)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [
                Assignment(
                    left = "_mat",
                    right = [10, 3, 5, 6]
                ),
                Assignment(
                    left = "_mat_len",
                    right = 4
                ),
                Assignment(
                    left = "_mat_idx",
                    right = 0
                ),
                Assignment(
                    left = "k",
                    right = 10
                )
            ],
            expr = Operator(op = "<", operands = ["_mat_idx", "_mat_len"]),
            body = [
                Call(
                    func = "disp",
                    arguments = ["k"]
                ),
                Assignment(
                    left = "_mat_idx",
                    right = Operator(
                        op = "+",
                        operands = ["_mat_idx", 1]
                    )
                ),
                Assignment(
                    left = "k",
                    right = "_mat[_mat_idx]"
               )
            ],
            post = []

        )
    )

