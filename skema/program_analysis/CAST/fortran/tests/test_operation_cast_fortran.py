import pytest
from tempfile import TemporaryDirectory
from pathlib import Path 

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    Operator,
    CASTLiteralValue,
    ScalarType
)

def binop1():
    return """
program binop1
integer :: x = 2 + 3
end program binop1
    """

def binop2():
    return """
program binop2
integer :: x = 2
integer :: y = x + 3
end program binop2
    """

def binop3():
    return """
program binop3
integer :: x = 1
integer :: y = 2
integer :: z = x + y - (y * x) / x
end program binop3
    """

def unary1():
    return """
program unary1
integer :: x = -1
end program unary1
    """

def unary2():
    return """
program unary2
integer :: x = 1
integer :: y = -x
end program unary2
    """


def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]

def test_binop1():
    exp_cast = generate_cast(binop1())
    # Test basic properties of assignment node
    binop_node = exp_cast.nodes[0].body[0]

    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "+"

    assert isinstance(binop_node.right.operands[0], CASTLiteralValue)
    assert binop_node.right.operands[0].value == '2'
    assert binop_node.right.operands[0].value_type == ScalarType.INTEGER

    assert isinstance(binop_node.right.operands[1], CASTLiteralValue)
    assert binop_node.right.operands[1].value == '3'
    assert binop_node.right.operands[1].value_type == ScalarType.INTEGER

def test_binop2():
    exp_cast = generate_cast(binop2())

    # Test basic properties of two assignment nodes
    binop_node = exp_cast.nodes[0].body[0]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"
    assert binop_node.left.val.id == 0

    assert isinstance(binop_node.right, CASTLiteralValue)
    assert binop_node.right.value_type == ScalarType.INTEGER
    assert binop_node.right.value == '2'
    
    # ------
    binop_node = exp_cast.nodes[0].body[1]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "y"
    assert binop_node.left.val.id == 1

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "+"

    assert isinstance(binop_node.right.operands[0], Name)
    assert binop_node.right.operands[0].name == 'x'
    assert binop_node.right.operands[0].id == 0

    assert isinstance(binop_node.right.operands[1], CASTLiteralValue)
    assert binop_node.right.operands[1].value == '3'
    assert binop_node.right.operands[1].value_type == ScalarType.INTEGER

def test_binop3():
    exp_cast = generate_cast(binop3())

    # Test basic properties of two assignment nodes
    binop_node = exp_cast.nodes[0].body[0]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"
    assert binop_node.left.val.id == 0

    assert isinstance(binop_node.right, CASTLiteralValue)
    assert binop_node.right.value_type == ScalarType.INTEGER
    assert binop_node.right.value == '1'
    
    # ------
    binop_node = exp_cast.nodes[0].body[1]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "y"
    assert binop_node.left.val.id == 1

    assert isinstance(binop_node.right, CASTLiteralValue)
    assert binop_node.right.value_type == ScalarType.INTEGER
    assert binop_node.right.value == '2'
    
    # ------
    binop_node = exp_cast.nodes[0].body[2]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "z"
    assert binop_node.left.val.id == 2

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "-"

    binop_node_1 = binop_node.right.operands[0]
    assert binop_node_1.op == "+"   

    assert isinstance(binop_node_1.operands[0], Name)
    assert binop_node_1.operands[0].name == "x"
    assert binop_node_1.operands[0].id == 0

    assert isinstance(binop_node_1.operands[1], Name)
    assert binop_node_1.operands[1].name == "y"
    assert binop_node_1.operands[1].id == 1

    binop_node_2 = binop_node.right.operands[1]
    assert binop_node_2.op == "/"   
    assert isinstance(binop_node_2.operands[0], Operator)

    assert isinstance(binop_node_2.operands[1], Name)
    assert binop_node_2.operands[1].name == "x"
    assert binop_node_2.operands[1].id == 0

    binop_node_3 = binop_node_2.operands[0]
    assert binop_node_3.op == "*"

    assert isinstance(binop_node_3.operands[0], Name)
    assert binop_node_3.operands[0].name == "y"
    assert binop_node_3.operands[0].id == 1

    assert isinstance(binop_node_3.operands[1], Name)
    assert binop_node_3.operands[1].name == "x"
    assert binop_node_3.operands[1].id == 0

def test_unary1():
    exp_cast = generate_cast(unary1())
    # Test basic properties of assignment node
    unary_node = exp_cast.nodes[0].body[0]

    assert isinstance(unary_node, Assignment)
    assert isinstance(unary_node.left, Var)
    assert isinstance(unary_node.left.val, Name)
    assert unary_node.left.val.name == "x"
    assert unary_node.left.val.id == 0

    assert isinstance(unary_node.right, Operator)
    assert unary_node.right.op == "-"

    assert isinstance(unary_node.right.operands[0], CASTLiteralValue)
    assert unary_node.right.operands[0].value == '1'
    assert unary_node.right.operands[0].value_type == ScalarType.INTEGER

def test_unary2():
    exp_cast = generate_cast(unary2())
    # Test basic properties of assignment node
    unary_node = exp_cast.nodes[0].body[0]

    assert isinstance(unary_node, Assignment)
    assert isinstance(unary_node.left, Var)
    assert isinstance(unary_node.left.val, Name)
    assert unary_node.left.val.name == "x"
    assert unary_node.left.val.id == 0

    assert isinstance(unary_node.right, CASTLiteralValue)
    assert unary_node.right.value == '1'
    assert unary_node.right.value_type == ScalarType.INTEGER

    unary_node = exp_cast.nodes[0].body[1]

    assert isinstance(unary_node, Assignment)
    assert isinstance(unary_node.left, Var)
    assert isinstance(unary_node.left.val, Name)
    assert unary_node.left.val.name == "y"
    assert unary_node.left.val.id == 1

    assert isinstance(unary_node.right, Operator)
    assert unary_node.right.op == "-"

    assert isinstance(unary_node.right.operands[0], Name)
    assert unary_node.right.operands[0].name == 'x'
    assert unary_node.right.operands[0].id == 0
