from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Call,
    LiteralValue,
    Operator,
    Module,
    Name,
    Var
)

def assert_call(call, func = None, arguments = None):
    """ Test the call for correct type, name, and arguments. """
    assert isinstance(call, Call)
    if func:
        assert_identifier(call.func, func)
    if arguments:
        for i, argument in enumerate(call.arguments):
            assert_operand(argument, arguments[i])

def assert_name(node, name = None):
    """ Test the node for correct type and name """
    assert isinstance(node, Name)
    if name:
        assert(node.name == name)

def assert_identifier(var, name = None):
    """ Test the Var for correct type and name. """
    assert isinstance(var, Var)
    if name:
        assert_name(var.val, name)

def assert_literal_value(literal_value, value = None):
    """ Test the LiteralValue for correct type and value. """
    assert isinstance(literal_value, LiteralValue)
    if value:
        assert literal_value.value == value

def assert_operand(operand, value = None):
    """ Test a Var or LiteralValue operand for correct type and value. """
    if value:
        if isinstance(operand, Var):
            assert_identifier(operand, name = value)
        elif isinstance(operand, LiteralValue):
            assert_literal_value(operand, value = value)
        else: 
            assert(False)

def assert_assignment(assignment, left = None, right = None):
    """ Test an Assignment for correct type and operands. """
    assert isinstance(assignment, Assignment)
    if left:
        assert_operand(assignment.left, left)
    if right:
        assert_operand(assignment.right, right)

def assert_expression(expression, op = None, operands = None):
    """ Test an Operator for correct type, operation, and operands. """
    assert isinstance(expression, Operator)
    if op:
        assert expression.op == op
    if operands:
        assert len(expression.operands) == len(operands)
        for i, operand in enumerate(expression.operands):
            assert_operand(operand, operands[i])

def cast_nodes(source):
    """ Return the CAST nodes from the first Module of MatlabToCast output """
    # there should only be one CAST object in the cast output list
    cast = MatlabToCast(source = source).out_cast
    assert len(cast) == 1
    # there should be one module in the CAST object
    assert len(cast[0].nodes) == 1
    module = cast[0].nodes[0]
    assert isinstance(module, Module)
    # return the module body node list
    return module.body
