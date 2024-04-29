import pytest
from tempfile import TemporaryDirectory
from pathlib import Path 

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    CASTLiteralValue,
    ModelIf,
    Operator,
    ScalarType
)

def cond1():
    return """
program cond1
    implicit none
    integer :: x = 2

    if (x < 5) then
        x = x + 1
    else
        x = x - 3
    end if
end program cond1
"""

def cond2():
    return """
program cond2
    implicit none
    integer :: x=2, y=3

    if (x < 5) then
        x = 1
        y = 2
        x = x * y
    else
        x = x - 3
    end if
end program cond2
"""

def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]

def test_cond1():
    exp_cast = generate_cast(cond1())
    
    # Test basic conditional
    asg_node = exp_cast.nodes[0].body[0]
    cond_node = exp_cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == ScalarType.INTEGER
    assert asg_node.right.value == '2'

    assert isinstance(cond_node, ModelIf)
    cond_expr = cond_node.expr
    cond_body = cond_node.body
    cond_else = cond_node.orelse

    assert isinstance(cond_expr, Operator)
    assert cond_expr.op == "<"
    assert isinstance(cond_expr.operands[0], Name)
    assert isinstance(cond_expr.operands[1], CASTLiteralValue)

    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert isinstance(cond_body[0].right, Operator)
    assert cond_body[0].right.op == "+" 

    assert len(cond_else) == 1
    assert isinstance(cond_else[0], Assignment)
    assert isinstance(cond_else[0].left, Var)
    assert isinstance(cond_else[0].right, Operator)
    assert cond_else[0].right.op == "-" 


def test_cond2():
    exp_cast = generate_cast(cond2())

    # Test multiple variable conditional
    asg_node = exp_cast.nodes[0].body[0]
    cond_node = exp_cast.nodes[0].body[2]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == ScalarType.INTEGER
    assert asg_node.right.value == '2'
    
    asg_node = exp_cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "y"
    assert asg_node.left.val.id == 1

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == ScalarType.INTEGER
    assert asg_node.right.value == '3'

    assert isinstance(cond_node, ModelIf)
    cond_expr = cond_node.expr
    cond_body = cond_node.body
    cond_else = cond_node.orelse

    assert isinstance(cond_expr, Operator)
    assert cond_expr.op == "<"
    assert isinstance(cond_expr.operands[0], Name)
    assert cond_expr.operands[0].name == "x"
    assert isinstance(cond_expr.operands[1], CASTLiteralValue)
    assert cond_expr.operands[1].value_type == ScalarType.INTEGER
    assert cond_expr.operands[1].value == "5"

    assert len(cond_body) == 3
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "x"
    assert isinstance(cond_body[0].right, CASTLiteralValue)
    assert cond_body[0].right.value == "1"

    assert isinstance(cond_body[1], Assignment)
    assert isinstance(cond_body[1].left, Var)
    assert cond_body[1].left.val.name == "y"
    assert isinstance(cond_body[1].right, CASTLiteralValue)
    assert cond_body[1].right.value == "2"

    assert isinstance(cond_body[2], Assignment)
    assert isinstance(cond_body[2].left, Var)
    assert isinstance(cond_body[2].right, Operator)

    assert cond_body[2].right.op == "*" 

    assert isinstance(cond_body[2].right.operands[0], Name)
    assert cond_body[2].right.operands[0].name == "x"
    assert cond_body[2].right.operands[0].id == 0
    assert isinstance(cond_body[2].right.operands[1], Name)
    assert cond_body[2].right.operands[1].name == "y"
    assert cond_body[2].right.operands[1].id == 1

    assert len(cond_else) == 1
    assert isinstance(cond_else[0], Assignment)
    assert isinstance(cond_else[0].left, Var)
    assert isinstance(cond_else[0].right, Operator)
    assert cond_else[0].right.op == "-" 