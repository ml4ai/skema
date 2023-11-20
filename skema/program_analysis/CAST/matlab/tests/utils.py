from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from typing import List
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    AstNode,
    Call,
    FunctionDef,
    LiteralValue,
    Loop,
    Operator,
    ModelIf,
    Module,
    Name,
    Var
)

def check(result, expected = None):
    """ Test for match with the same datatypes. """
    if isinstance(result, List):
        assert len(result) == len(expected)
        for i, _ in enumerate(result):
            check(_, expected[i])
    elif isinstance(result, Assignment):
        check(result.left, expected.left)
        check(result.right, expected.right)
    elif isinstance(result, Operator):
        check(result.op, expected.op)
        check(result.operands, expected.operands)
    elif isinstance(result, Call):
        check(result.func, expected.func)
        check(result.arguments, expected.arguments)
    elif isinstance(result, FunctionDef):
        check(result.name, expected.name)
        check(result.func_args, expected.func_args)
        check(result.body, expected.body)
    elif isinstance(result, ModelIf):
        check(result.expr, expected.expr)
        check(result.body, expected.body)
        check(result.orelse, expected.orelse)
    elif isinstance(result, Loop):
        check(result.pre, expected.pre)
        check(result.expr, expected.expr)
        check(result.body, expected.body)
        check(result.post, expected.post)
    elif isinstance(result, LiteralValue):
        check(result.value, expected)
    elif isinstance(result, Var):
        check(result.val, expected)
    elif isinstance(result, Name):
        check(result.name, expected)
    else:
        assert result == expected

    # every CAST node has a source_refs element
    if isinstance(result, AstNode):
        assert not result.source_refs == None

# we curently produce a CAST object with a single Module in the nodes list.
def cast(source):
    """ Return the MatlabToCast output """
    # there should only be one CAST object in the cast output list
    cast = MatlabToCast(source = source).out_cast
    # there should be one module in the CAST object
    assert len(cast.nodes) == 1
    module = cast.nodes[0]
    assert isinstance(module, Module)
    # return the module body node list
    return module.body
