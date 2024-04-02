# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Name,
    CASTLiteralValue,
    ModelIf,
    Operator
)

def cond_compound1():
    return """
a = 3

if 1 < a < 10:
    a = 40
    """

def cond_compound2():
    return """
a = 3

if 1 < a < 10 < 20:
    a = 40
    """

def cond_compound3():
    return """
a = 3

if ((1 < a) or (a < 10)) and (a != 5):
    a = 40
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_cond_compound1():
    exp_cast = generate_cast(cond_compound1())
    
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "a"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'

    assert isinstance(cond_node, ModelIf)
    cond_node = exp_cast.nodes[0].body[1]
    cond_expr = cond_node.expr
    assert isinstance(cond_expr, ModelIf)

    def check_inner_if(if_node):
        assert isinstance(if_node, ModelIf)

        assert len(if_node.expr) == 1
        expr = if_node.expr[0]
        assert isinstance(expr, Operator)
        assert expr.op == "ast.Lt"
        assert len(expr.operands) == 2
        assert isinstance(expr.operands[0], CASTLiteralValue)
        assert expr.operands[0].value_type == "Integer"
        assert expr.operands[0].value == "1"

        assert isinstance(expr.operands[1], Name)
        assert expr.operands[1].name == "a"
        assert expr.operands[1].id == 0

        assert len(if_node.body) == 1
        body = if_node.body[0]
        assert isinstance(body, Operator)
        assert body.op == "ast.Lt"
        assert len(body.operands) == 2
        assert isinstance(body.operands[0], Name)
        assert body.operands[0].name == "a"
        assert body.operands[0].id == 0

        assert isinstance(body.operands[1], CASTLiteralValue)
        assert body.operands[1].value_type == "Integer"
        assert body.operands[1].value == "10"

        assert len(if_node.orelse) == 1
        orelse = if_node.orelse[0]
        assert isinstance(orelse, CASTLiteralValue)
        assert orelse.value_type == "Boolean"
        assert orelse.value == "False"

    check_inner_if(cond_expr)

    cond_body = cond_node.body
    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "a"
    assert cond_body[0].left.val.id == 0

    assert isinstance(cond_body[0].right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '40'

    cond_else = cond_node.orelse
    assert len(cond_else) == 0

def test_cond_compound2():
    exp_cast = generate_cast(cond_compound2())
    
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "a"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'

    assert isinstance(cond_node, ModelIf)
    cond_node = exp_cast.nodes[0].body[1]
    cond_expr = cond_node.expr
    assert isinstance(cond_expr, ModelIf)

    def check_inner_inner_if(if_node):
        assert isinstance(if_node, ModelIf)

        assert len(if_node.expr) == 1
        expr = if_node.expr[0]
        assert isinstance(expr, Operator)
        assert expr.op == "ast.Lt"
        assert len(expr.operands) == 2
        assert isinstance(expr.operands[0], Name)
        assert expr.operands[0].name == "a"
        assert expr.operands[0].id == 0

        assert isinstance(expr.operands[1], CASTLiteralValue)
        assert expr.operands[1].value_type == "Integer"
        assert expr.operands[1].value == "10"

        assert len(if_node.body) == 1
        body = if_node.body[0]
        assert isinstance(body, Operator)
        assert body.op == "ast.Lt"
        assert len(body.operands) == 2
        assert isinstance(body.operands[0], CASTLiteralValue)
        assert body.operands[0].value_type == "Integer"
        assert body.operands[0].value == "10"

        assert isinstance(body.operands[1], CASTLiteralValue)
        assert body.operands[1].value_type == "Integer"
        assert body.operands[1].value == "20"

        assert len(if_node.orelse) == 1
        orelse = if_node.orelse[0]
        assert isinstance(orelse, CASTLiteralValue)
        assert orelse.value_type == "Boolean"
        assert orelse.value == "False"

    def check_inner_if(if_node):
        assert isinstance(if_node, ModelIf)

        assert len(if_node.expr) == 1
        expr = if_node.expr[0]
        assert isinstance(expr, Operator)
        assert expr.op == "ast.Lt"
        assert len(expr.operands) == 2
        assert isinstance(expr.operands[0], CASTLiteralValue)
        assert expr.operands[0].value_type == "Integer"
        assert expr.operands[0].value == "1"

        assert isinstance(expr.operands[1], Name)
        assert expr.operands[1].name == "a"
        assert expr.operands[1].id == 0

        assert len(if_node.body) == 1
        body = if_node.body[0]
        assert isinstance(body, ModelIf)
        check_inner_inner_if(body)        

        assert len(if_node.orelse) == 1
        orelse = if_node.orelse[0]
        assert isinstance(orelse, CASTLiteralValue)
        assert orelse.value_type == "Boolean"
        assert orelse.value == "False"

    check_inner_if(cond_expr)

    cond_body = cond_node.body
    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "a"
    assert cond_body[0].left.val.id == 0

    assert isinstance(cond_body[0].right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '40'

    cond_else = cond_node.orelse
    assert len(cond_else) == 0

def test_cond_compound3():
    exp_cast = generate_cast(cond_compound3())
    
    asg_node = exp_cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "a"
    assert asg_node.left.val.id == 0

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '3'

    assert isinstance(cond_node, ModelIf)
    cond_node = exp_cast.nodes[0].body[1]
    cond_expr = cond_node.expr
    assert isinstance(cond_expr, ModelIf)

    def check_inner_inner_if(if_node):
        assert isinstance(if_node, ModelIf)

        assert len(if_node.expr) == 1
        expr = if_node.expr[0]
        assert isinstance(expr, Operator)
        assert expr.op == "ast.Lt"
        assert len(expr.operands) == 2

        assert isinstance(expr.operands[0], CASTLiteralValue)
        assert expr.operands[0].value_type == "Integer"
        assert expr.operands[0].value == "1"

        assert isinstance(expr.operands[1], Name)
        assert expr.operands[1].name == "a"
        assert expr.operands[1].id == 0

        assert len(if_node.body) == 1
        body = if_node.body[0]
        assert isinstance(body, CASTLiteralValue)
        assert body.operands[0].value_type == "Boolean"
        assert body.operands[0].value == "True"

        assert len(if_node.orelse) == 1
        orelse = if_node.orelse[0]
        assert isinstance(orelse, Operator)
        assert orelse.op == "ast.Lt"
        assert len(orelse.operands) == 2

        assert isinstance(orelse.operands[0], Name)
        assert orelse.operands[0].name == "a"
        assert orelse.operands[0].id == 0

        assert isinstance(orelse.operands[1], CASTLiteralValue)
        assert orelse.operands[1].value_type == "Integer"
        assert orelse.operands[1].value == "10"

    def check_inner_if(if_node):
        assert isinstance(if_node, ModelIf)
        
        assert len(if_node.expr) == 1
        expr = if_node.expr[0]
        assert isinstance(expr, ModelIf)
        check_inner_inner_if(expr)        

        assert len(if_node.body) == 1
        body = if_node.body[0]
        assert isinstance(body, Operator)
        assert body.op == "ast.NotEq"
        assert len(body.operands) == 2
        assert isinstance(body.operands[0], Name)
        assert body.operands[0].name == "a"
        assert body.operands[0].id == 0

        assert isinstance(body.operands[1], CASTLiteralValue)
        assert body.operands[1].value_type == "Integer"
        assert body.operands[1].value == "5"

        assert len(if_node.orelse) == 1
        orelse = if_node.orelse[0]
        assert isinstance(orelse, CASTLiteralValue)
        assert orelse.value_type == "Boolean"
        assert orelse.value == "False"

    check_inner_if(cond_expr)

    cond_body = cond_node.body
    assert len(cond_body) == 1
    assert isinstance(cond_body[0], Assignment)
    assert isinstance(cond_body[0].left, Var)
    assert cond_body[0].left.val.name == "a"
    assert cond_body[0].left.val.id == 0

    assert isinstance(cond_body[0].right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '40'

    cond_else = cond_node.orelse
    assert len(cond_else) == 0





# def test_cond2():
#     exp_cast = generate_cast(cond2())

#     # Test multiple variable conditional
#     asg_node = exp_cast.nodes[0].body[0]
#     cond_node = exp_cast.nodes[0].body[2]

#     assert isinstance(asg_node, Assignment)
#     assert isinstance(asg_node.left, Var)
#     assert isinstance(asg_node.left.val, Name)
#     assert asg_node.left.val.name == "x"
#     assert asg_node.left.val.id == 0

#     assert isinstance(asg_node.right, CASTLiteralValue)
#     assert asg_node.right.value_type == "Integer"
#     assert asg_node.right.value == '2'
    
#     asg_node = exp_cast.nodes[0].body[1]
#     assert isinstance(asg_node, Assignment)
#     assert isinstance(asg_node.left, Var)
#     assert isinstance(asg_node.left.val, Name)
#     assert asg_node.left.val.name == "y"
#     assert asg_node.left.val.id == 1

#     assert isinstance(asg_node.right, CASTLiteralValue)
#     assert asg_node.right.value_type == "Integer"
#     assert asg_node.right.value == '3'

#     assert isinstance(cond_node, ModelIf)
#     cond_expr = cond_node.expr
#     cond_body = cond_node.body
#     cond_else = cond_node.orelse

#     assert isinstance(cond_expr, Operator)
#     assert cond_expr.op == "ast.Lt"
#     assert isinstance(cond_expr.operands[0], Name)
#     assert cond_expr.operands[0].name == "x"
#     assert isinstance(cond_expr.operands[1], CASTLiteralValue)
#     assert cond_expr.operands[1].value_type == "Integer"
#     assert cond_expr.operands[1].value == "5"

#     assert len(cond_body) == 3
#     assert isinstance(cond_body[0], Assignment)
#     assert isinstance(cond_body[0].left, Var)
#     assert cond_body[0].left.val.name == "x"
#     assert isinstance(cond_body[0].right, CASTLiteralValue)
#     assert cond_body[0].right.value == "1"

#     assert isinstance(cond_body[1], Assignment)
#     assert isinstance(cond_body[1].left, Var)
#     assert cond_body[1].left.val.name == "y"
#     assert isinstance(cond_body[1].right, CASTLiteralValue)
#     assert cond_body[1].right.value == "2"

#     assert isinstance(cond_body[2], Assignment)
#     assert isinstance(cond_body[2].left, Var)
#     assert isinstance(cond_body[2].right, Operator)

#     assert cond_body[2].right.op == "ast.Mult" 

#     assert isinstance(cond_body[2].right.operands[0], Name)
#     assert cond_body[2].right.operands[0].name == "x"
#     assert cond_body[2].right.operands[0].id == 0
#     assert isinstance(cond_body[2].right.operands[1], Name)
#     assert cond_body[2].right.operands[1].name == "y"
#     assert cond_body[2].right.operands[1].id == 1

#     assert len(cond_else) == 1
#     assert isinstance(cond_else[0], Assignment)
#     assert isinstance(cond_else[0].left, Var)
#     assert isinstance(cond_else[0].right, Operator)
#     assert cond_else[0].right.op == "ast.Sub" 

# def test_cond3():
#     exp_cast = generate_cast(cond3())

#     # Test nested ifs
#     cond_node = exp_cast.nodes[0].body[2]

#     assert isinstance(cond_node, ModelIf)
#     cond_body = cond_node.body
#     cond_else = cond_node.orelse

#     assert len(cond_body) == 2
#     assert len(cond_else) == 1
#     assert isinstance(cond_else[0], ModelIf)
#     nested_if = cond_else[0]
#     cond_body = nested_if.body
#     cond_else = nested_if.orelse

#     assert len(cond_body) == 2
#     assert len(cond_else) == 1
#     assert isinstance(cond_else[0], ModelIf)
#     nested_if = cond_else[0]
#     cond_body = nested_if.body
#     cond_else = nested_if.orelse

#     assert len(cond_body) == 3
#     assert len(cond_else) == 2
#     assert isinstance(cond_else[0], Assignment)
#     assert isinstance(cond_else[1], Assignment)
