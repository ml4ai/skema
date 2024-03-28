import pytest
from tempfile import TemporaryDirectory
from pathlib import Path 

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    CASTLiteralValue
)

def exp0():
    return """
program exp0
integer :: x = 2
end program exp0
    """

def exp1():
    return """
program exp1
integer :: x = 2
integer :: y = 3
end program exp1
"""

def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]

def test_exp0():
    exp_cast = generate_cast(exp0())
    # Test basic properties of assignment node
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'


def test_exp1():
    exp_cast = generate_cast(exp1())

    # Test basic properties of two assignment nodes
    asg_node = exp_cast.nodes[0].body[0]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'
    
    # ------
    asg_node = exp_cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "y"
    assert asg_node.left.val.id == 1

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'

if __name__ == "__main__": 
    cast = generate_cast(exp0())