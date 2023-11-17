from skema.program_analysis.CAST.matlab.tests.utils import (
    check,
    cast_nodes
)
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
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(op = "==", operands = ["x", 5]),
            # then
            body = [Assignment(left="y", right = 6)]
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
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(op = ">", operands = ["x", 5]),
            # then
            body = [
                Assignment(left="y", right = 6),
                Assignment(left="three", right = 3)
            ],
            # else
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
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(op = ">=", operands = ["x", 5]),
            # then
            body = [
                Assignment(left="y", right = 6)
            ],
            # else
            orelse = [
                ModelIf(
                    # if
                    expr = Operator(op = "<=", operands = ["x", 0]),
                    # then
                    body = [
                        Assignment(left="y", right = "x")
                    ]   
                )   
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
        cast_nodes(source)[0],
        ModelIf(
            # if
            expr = Operator(op = ">", operands = ["x", 5]),
            # then
            body = [
                Assignment(left="a", right = 6)
            ],
            # else
            orelse = [
                ModelIf(
                    # if
                    expr = Operator(op = ">", operands = ["x", 0]),
                    # then
                    body = [
                        Assignment(left="b", right = "x")
                    ],
                    orelse = [
                        Assignment(left="c", right = 0)
                    ]
                )
            ]
        )
    )
