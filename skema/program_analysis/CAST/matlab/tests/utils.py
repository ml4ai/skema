from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from typing import List
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    AstNode,
    Call,
    LiteralValue,
    Loop,
    Operator,
    Module,
    Name,
    Var
)

def assert_assignment(assignment, left = None, right = None):
    """ Test an Assignment for correct type and operands. """
    assert isinstance(assignment, Assignment)
    assert not assignment.source_refs == None
    assert_value(assignment.left, left)
    assert_value(assignment.right, right)

def assert_call(call, func = None, arguments = None):
    """ Test a call for correct type, function, and arguments. """
    assert isinstance(call, Call)
    assert not call.source_refs == None
    assert_value(call.func, func)
    assert_value(call.arguments, arguments)

def assert_operator(operator, op = None, operands = None):
    """ Test an Operator for correct type, operation, and operands. """
    assert isinstance(operator, Operator)
    assert not operator.source_refs == None
    assert operator.op == op
    assert_value(operator.operands, operands)

def assert_loop(loop, pre = None, expr = None, body = None, post = None):
    """ Test a Loop for correct type and fields. """
    assert isinstance(loop, Loop)
    assert not loop.source_refs == None
    assert_value(loop.pre, pre)
    assert_value(loop.expr, expr)
    assert_value(loop.body, body)
    assert_value(loop.post, post)

def assert_value(operand, value = None):
    """ Test for equality. """
    if value:
        if isinstance(operand, List):
            for i, element in enumerate(operand):
                assert_value(element, value[i])
        elif isinstance(operand, Var):
            assert_value(operand.val, value)
        elif isinstance(operand, Name):
            assert_value(operand.name, value)
        elif isinstance(operand, LiteralValue):
            assert_value(operand.value, value)
        else:
            assert operand == value

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
