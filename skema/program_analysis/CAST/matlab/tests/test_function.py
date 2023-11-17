from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    Operator
)

# Test CAST from functions
def no_test_function_definition():
    """ Test function definition """
    source = """
    function both = add_them(x, y)
        both = x + y
    end
    """
    nodes = cast(source)
    assert len(nodes) == 1

def test_literal_args():
    """ Test function call with literal arguments """
    nodes = cast("x = both(3, 5)")
    check(
        nodes[0],
        Assignment(
            left = "x",
            right = Call(
                func = "both", 
                arguments = [3, 5]
            )
        )
    )

def test_inline_operator_args():
    """ Test function call with Operator arguments """
    nodes = cast("foo(x < a, -6)")
    check(
        nodes[0],
        Call(
            func = "foo",
            arguments = [
                Operator (
                    op = "<",
                    operands = ["x", "a"]
                ),
                Operator (
                    op = "-",
                    operands = [6]
                ),
            ]
        )
    )

def test_nested_calls():
    """ Test function call with matrix of function call arguments """
    nodes = cast("foo(bar(x), baz(y))")
    check(
        nodes[0],
        Call(
            func = "foo",
            arguments = [
                Call (
                    func = "bar",
                    arguments = ["x"]
                ),
                Call (
                    func = "baz",
                    arguments = ["y"]
                ),
            ]
        )
    )
