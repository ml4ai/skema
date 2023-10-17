from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue,
    Name,
    Var
)

# Test the CAST returned by processing the simplest MATLAB assignment

def test_assignment():
    """ Tests parser assignment CAST """

    source = 'x = 5;'
    
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
    assert left.val.name == "x"

    # Right branch of this assignment is an Integer literal
    right = node.right
    assert isinstance(right, LiteralValue)
    assert right.value_type == "Integer"
    assert right.value == "5"
