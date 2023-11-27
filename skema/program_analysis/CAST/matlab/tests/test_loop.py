from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    Loop,
    Operator
)

# Test the for loop
def test_for():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 1:10
            x = do_something(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            # pre = [Assignment(left = "n", right = 1)],
            # expr = Operator(op = "<", operands = ["n", 10]),
            pre = [],
            expr = None,
            body = [
                Assignment(
                    left = "x",
                    right = Call(
                        func = "do_something",
                        arguments = ["n"]
                    )
                ),
                Call(
                    func = "disp",
                    arguments = ["k"]
                )
            ],
            post = []
        )
    )


# Test the for loop
def test_while():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        while k < 5
            k = k + 2;
            disp(k);
        end
    """
    nodes = cast(source)
    check(nodes[0], 
        Loop(
            pre = [],
            expr = Operator(op = "<", operands = ["k",5]),
            body = [
                Assignment(
                    left = "k",
                    right = Operator(op = "+", operands = ["k",2])
                ),
                Call(
                    func = "disp",
                    arguments = ["k"]
                )
            ],
            post = []
        )
    )

