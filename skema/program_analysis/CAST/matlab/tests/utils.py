from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from typing import List
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    AstNode,
    Call,
    LiteralValue,
    Loop,
    Operator,
    ModelIf,
    Module,
    Name,
    Var
)

def assert_foo(result, expected = None):
    """ Test for equality. """
    if isinstance(result, Assignment):
        assert isinstance(expected, Assignment)
        assert_foo(result.left, expected.left)
        assert_foo(result.right, expected.right)
    elif isinstance(result, Operator):
        assert isinstance(expected, Operator)
        assert_foo(result.op, expected.op)
        assert_foo(result.operands, expected.operands)
    elif isinstance(result, Call):
        assert isinstance(expected, Call)
        assert_foo(result.func, expected.func)
        assert_foo(result.arguments, expected.arguments)
    elif isinstance(result, ModelIf):
        assert isinstance(expected, ModelIf)
        assert_foo(result.expr, expected.expr)
        assert_foo(result.body, expected.body)
    elif isinstance(result, LiteralValue):
        assert_foo(result.value, expected)
    elif isinstance(result, Var):
        assert_foo(result.val, expected)
    elif isinstance(result, Name):
        assert_foo(result.name, expected)

    elif isinstance(result, List):
        print("\nassert_foo with List")
        print(f"result = {result}")
        print(f"expected = {expected}")
        print(f"len(result) = {len(result)}")
        print(f"len(expected) = {len(expected)}")
        assert len(result) == len(expected)
        for i, element in enumerate(result):
            assert_foo(element, expected[i])
    else:
        assert result == expected

    if isinstance(result, AstNode):
        assert not result.source_refs == None


# we curently produce a CAST object with a single Module in the nodes list.
def cast_nodes(source):
    """ Return the MatlabToCast output """
    # there should only be one CAST object in the cast output list
    cast = MatlabToCast(source = source).out_cast
    # there should be one module in the CAST object
    assert len(cast.nodes) == 1
    module = cast.nodes[0]
    assert isinstance(module, Module)
    # return the module body node list
    return module.body
