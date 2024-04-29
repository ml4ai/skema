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

def cond_compound1():
    return """
program cond_compound1
integer :: a = 3
if (a .gt. 1 .and. a .lt. 10) then
    a = 40
end if
end program cond_compound1
    """


def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]

def test_cond_compound1():
    exp_cast = generate_cast(cond_compound1())
    
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "a"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == ScalarType.INTEGER
    assert asg_node.right.value == '3'

    cond_node = exp_cast.nodes[0].body[1]
    cond_expr = cond_node.expr
    assert isinstance(cond_node, ModelIf)
    assert isinstance(cond_expr, ModelIf)

    if_node = cond_expr
    assert isinstance(if_node, ModelIf)

    expr = if_node.expr
    assert isinstance(expr, Operator)
    assert expr.op == ".gt."
    assert len(expr.operands) == 2
    assert isinstance(expr.operands[1], CASTLiteralValue)
    assert expr.operands[1].value_type == ScalarType.INTEGER
    assert expr.operands[1].value == "1"

    assert isinstance(expr.operands[0], Name)
    assert expr.operands[0].name == "a"
    assert expr.operands[0].id == 0

    assert len(if_node.body) == 1
    body = if_node.body[0]
    assert isinstance(body, Operator)
    assert body.op == ".lt."
    assert len(body.operands) == 2
    assert isinstance(body.operands[0], Name)
    assert body.operands[0].name == "a"
    assert body.operands[0].id == 0

    assert isinstance(body.operands[1], CASTLiteralValue)
    assert body.operands[1].value_type == ScalarType.INTEGER
    assert body.operands[1].value == "10"

    assert len(if_node.orelse) == 1
    orelse = if_node.orelse[0]
    assert isinstance(orelse, CASTLiteralValue)
    assert orelse.value_type == ScalarType.BOOLEAN
    assert orelse.value == False

    cond_body = cond_node.body
    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "a"
    assert cond_body[0].left.val.id == 0

    assert isinstance(cond_body[0].right, CASTLiteralValue)
    assert cond_body[0].right.value_type == ScalarType.INTEGER
    assert cond_body[0].right.value == '40'

    cond_else = cond_node.orelse
    assert len(cond_else) == 0
