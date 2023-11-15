# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    Operator,
    LiteralValue
)

def binop1():
    return """
x = 2 + 3
    """

def binop2():
    return """
x = 2
y = x + 3
    """

def binop3():
    return """
x = 1 
y = 2
z = x + y - (y * x) / x
    """

def unary1():
    return """
x = -1 
    """

def unary2():
    return """
x = 1
y = -x 
    """


def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_binop1():
    exp_cast = generate_cast(binop1())
    # Test basic properties of assignment node
    binop_node = exp_cast.nodes[0].body[0]

    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "ast.Add"

    assert isinstance(binop_node.right.operands[0], LiteralValue)
    assert binop_node.right.operands[0].value == '2'
    assert binop_node.right.operands[0].value_type == 'Integer'

    assert isinstance(binop_node.right.operands[1], LiteralValue)
    assert binop_node.right.operands[1].value == '3'
    assert binop_node.right.operands[1].value_type == 'Integer'

def test_binop2():
    exp_cast = generate_cast(binop2())

    # Test basic properties of two assignment nodes
    binop_node = exp_cast.nodes[0].body[0]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"
    assert binop_node.left.val.id == 0

    assert isinstance(binop_node.right, LiteralValue)
    assert binop_node.right.value_type == "Integer"
    assert binop_node.right.value == '2'
    
    # ------
    binop_node = exp_cast.nodes[0].body[1]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "y"
    assert binop_node.left.val.id == 1

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "ast.Add"

    assert isinstance(binop_node.right.operands[0], Name)
    assert binop_node.right.operands[0].name == 'x'
    assert binop_node.right.operands[0].id == 0

    assert isinstance(binop_node.right.operands[1], LiteralValue)
    assert binop_node.right.operands[1].value == '3'
    assert binop_node.right.operands[1].value_type == 'Integer'

def test_binop3():
    exp_cast = generate_cast(binop3())

    # Test basic properties of two assignment nodes
    binop_node = exp_cast.nodes[0].body[0]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "x"
    assert binop_node.left.val.id == 0

    assert isinstance(binop_node.right, LiteralValue)
    assert binop_node.right.value_type == "Integer"
    assert binop_node.right.value == '1'
    
    # ------
    binop_node = exp_cast.nodes[0].body[1]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "y"
    assert binop_node.left.val.id == 1

    assert isinstance(binop_node.right, LiteralValue)
    assert binop_node.right.value_type == "Integer"
    assert binop_node.right.value == '2'
    
    # ------
    binop_node = exp_cast.nodes[0].body[2]
    assert isinstance(binop_node, Assignment)
    assert isinstance(binop_node.left, Var)
    assert isinstance(binop_node.left.val, Name)
    assert binop_node.left.val.name == "z"
    assert binop_node.left.val.id == 2

    assert isinstance(binop_node.right, Operator)
    assert binop_node.right.op == "ast.Sub"

    binop_node_1 = binop_node.right.operands[0]
    assert binop_node_1.op == "ast.Add"   

    assert isinstance(binop_node_1.operands[0], Name)
    assert binop_node_1.operands[0].name == "x"
    assert binop_node_1.operands[0].id == 0

    assert isinstance(binop_node_1.operands[1], Name)
    assert binop_node_1.operands[1].name == "y"
    assert binop_node_1.operands[1].id == 1

    binop_node_2 = binop_node.right.operands[1]
    assert binop_node_2.op == "ast.Div"   
    assert isinstance(binop_node_2.operands[0], Operator)

    assert isinstance(binop_node_2.operands[1], Name)
    assert binop_node_2.operands[1].name == "x"
    assert binop_node_2.operands[1].id == 0

    binop_node_3 = binop_node_2.operands[0]
    assert binop_node_3.op == "ast.Mult"

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
    assert unary_node.right.op == "ast.USub"

    assert isinstance(unary_node.right.operands[0], LiteralValue)
    assert unary_node.right.operands[0].value == '1'
    assert unary_node.right.operands[0].value_type == 'Integer'

def test_unary2():
    exp_cast = generate_cast(unary2())
    # Test basic properties of assignment node
    unary_node = exp_cast.nodes[0].body[0]

    assert isinstance(unary_node, Assignment)
    assert isinstance(unary_node.left, Var)
    assert isinstance(unary_node.left.val, Name)
    assert unary_node.left.val.name == "x"
    assert unary_node.left.val.id == 0

    assert isinstance(unary_node.right, LiteralValue)
    assert unary_node.right.value == '1'
    assert unary_node.right.value_type == 'Integer'

    unary_node = exp_cast.nodes[0].body[1]

    assert isinstance(unary_node, Assignment)
    assert isinstance(unary_node.left, Var)
    assert isinstance(unary_node.left.val, Name)
    assert unary_node.left.val.name == "y"
    assert unary_node.left.val.id == 1

    assert isinstance(unary_node.right, Operator)
    assert unary_node.right.op == "ast.USub"

    assert isinstance(unary_node.right.operands[0], Name)
    assert unary_node.right.operands[0].name == 'x'
    assert unary_node.right.operands[0].id == 0
