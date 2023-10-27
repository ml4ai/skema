from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue,
    Name,
    Operator,
    Var
)

# Test the CAST returned by processing the simplest MATLAB binary assignment

def test_binary_operation():
    """ Tests parser binary operation CAST """

    source = 'z = x + y;'

    cast = MatlabToCast(source = source).out_cast

    # there should only be one CAST object in the cast output list
    assert len(cast) == 1

    # The root of the CAST should be assignment
    node = cast[0].nodes[0].body[0]
    assert isinstance(node, Assignment)

    # Left branch of an assignment node is the variable
    left = node.left
    assert isinstance(left, Var)
    assert isinstance(left.val, Name)
    assert left.val.name == "z"

    # right branch of this assignment node is a binary operator
    right = node.right
    assert right.op == "+"
    assert isinstance(right, Operator)
    assert isinstance(right.operands[0], Var)
    assert isinstance(right.operands[0].val, Name)
    assert right.operands[0].val.name == "x"
    assert isinstance(right.operands[1], Var)
    assert isinstance(right.operands[1].val, Name)
    assert right.operands[1].val.name == "y"
