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
    if left:
        assert_operand(assignment.left, left)
    if right:
        assert_operand(assignment.right, right)

def assert_call(call, func = None, arguments = None):
    """ Test a call for correct type, function, and arguments. """
    assert isinstance(call, Call)
    assert not call.source_refs == None
    if func:
        assert_operand(call.func, func)
    if arguments:
        assert_operand(call.arguments, arguments)

def assert_expression(expression, op = None, operands = None):
    """ Test an Operator for correct type, operation, and operands. """
    assert isinstance(expression, Operator)
    assert not expression.source_refs == None
    if op:
        assert expression.op == op
    if operands:
        assert len(expression.operands) == len(operands)
        for i, operand in enumerate(expression.operands):
            assert_operand(operand, operands[i])
        # assert_operand(expression.operands, operands)

def assert_loop(loop, pre = None, expr = None, body = None, post = None):
    """ Test a Loop for correct type and fields. """
    assert isinstance(loop, Loop)
    assert not loop.source_refs == None
    if pre:
        assert_operand(loop.pre, pre)
    if expr:
        assert_operand(loop.expr, expr)
    if body:
        assert_operand(loop.body, body)
    if post:
        assert_operand(loop.post, post)

def assert_operand(operand, value = None):
    """ Test for equality. """
    if value:
        if isinstance(operand, List):
            for i, element in enumerate(operand):
                assert_operand(element, value[i])
        elif isinstance(operand, Var):
            assert_operand(operand.val, value)
        elif isinstance(operand, Name):
            assert_operand(operand.name, value)
        elif isinstance(operand, LiteralValue):
            assert_operand(operand.value, value)
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
