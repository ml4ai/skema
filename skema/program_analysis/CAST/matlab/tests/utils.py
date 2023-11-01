from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue,
    Operator,
    Module,
    Name,
    Var
)

def assert_var(var, name = ""):
    """ Test the Var for correct type and name. """
    assert isinstance(var, Var)
    assert isinstance(var.val, Name)
    assert var.val.name == name

def assert_literal_value(literal_value, value = ""):
    """ Test the LiteralValue for correct type and value. """
    assert isinstance(literal_value, LiteralValue)
    assert literal_value.value == value

def assert_operand(operand, value = ""):
    """ Test a Var or LiteralValue operand for correct type and value. """
    if isinstance(operand, Var):
        assert_var(operand, value)
    elif isinstance(operand, LiteralValue):
        assert_literal_value(operand, value)
    else:
        assert(False)

def assert_assignment(assignment, left = "", right = ""):
    """ Test an Assignment correct type and operands. """
    assert isinstance(assignment, Assignment)
    assert_operand(assignment.left, left)
    assert_operand(assignment.right, right)

def assert_expression(expression, op = "", left = "", right = ""):
    """ Test an Operator for correct type, operation, and operands. """
    assert isinstance(expression, Operator)
    assert expression.op == op
    assert_operand(expression.operands[0], left)
    assert_operand(expression.operands[1], right)

def first_cast_node(source):
    """ Return the first node from the first Module of MatlabToCast output """

    # there should only be one CAST object in the cast output list
    cast = MatlabToCast(source = source).out_cast
    assert len(cast) == 1

    # there should be one module in the CAST object
    assert len(cast[0].nodes) == 1
    module = cast[0].nodes[0]
    assert isinstance(module, Module)

    # currently we support one node per module.  This may change
    assert len(module.body) == 1
    return module.body[0]
