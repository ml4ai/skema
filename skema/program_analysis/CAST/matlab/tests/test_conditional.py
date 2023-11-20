from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    ModelIf,
    Operator
)

def test_if():
    """ Test CAST from MATLAB 'if' conditional logic."""
    source = """
    if x == 5
        y = 6
    end
    """
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(op = "==", operands = ["x", 5]),
            body = [Assignment(left="y", right = 6)],
            orelse = []
        )
    )

def test_if_else():
    """  Test CAST from MATLAB 'if else' conditional logic."""
    source = """
    if x > 5
        y = 6
        three = 3
    else
        y = x
        foo = 'bar'
    end
    """
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(op = ">", operands = ["x", 5]),
            body = [
                Assignment(left="y", right = 6),
                Assignment(left="three", right = 3)
            ],
            orelse = [
                Assignment(left="y", right = "x"),
                Assignment(left="foo", right = "'bar'")
            ]
        )
    )


def test_if_elseif():
    """ Test CAST from MATLAB 'if elseif else' conditional logic."""
    source = """
    if x >= 5
        y = 6
    elseif x <= 0
        y = x
    end
    """
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(op = ">=", operands = ["x", 5]),
            body = [Assignment(left="y", right = 6)],
            orelse = [ModelIf(
                expr = Operator(op = "<=", operands = ["x", 0]),
                body = [Assignment(left="y", right = "x")],
                orelse = [])
            ]
        )
    )
    

def test_if_elseif_else():
    """ Test CAST from MATLAB 'if elseif else' conditional logic."""
    source = """
    if x > 5
        a = 6
    elseif x > 0
        b = x
    else
        c = 0
    end
    """
    check(
        cast(source)[0],
        ModelIf(
            expr = Operator(op = ">", operands = ["x", 5]),
            body = [Assignment(left="a", right = 6)],
            orelse = [
                ModelIf(
                    expr = Operator(op = ">", operands = ["x", 0]),
                    body = [Assignment(left="b", right = "x")],
                    orelse = [Assignment(left="c", right = 0)]
                )
            ]
        )
    )
