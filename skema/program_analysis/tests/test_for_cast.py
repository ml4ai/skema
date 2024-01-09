# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Call,
    Name,
    LiteralValue,
    ModelIf,
    Loop,
    Operator
)

def for1():
    return """
x = 7
for i in range(10):
    x = x + i
    """

def for2():
    return """
x = 1
for a,b in range(10):
    x = x + a + b
    """

def for3():
    return """
x = 1
L = [1,2,3]

for i in L:
    x = x + i
    """


def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_for1():
    cast = generate_cast(for1())

    asg_node = cast.nodes[0].body[0]
    loop_node = cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '7'

    assert isinstance(loop_node, Loop)
    assert len(loop_node.pre) == 2

    # Loop Pre
    loop_pre = loop_node.pre
    assert isinstance(loop_pre[0], Assignment)
    assert isinstance(loop_pre[0].left, Var)
    assert loop_pre[0].left.val.name == "generated_iter_0"

    assert isinstance(loop_pre[0].right, Call)
    assert loop_pre[0].right.func.name == "iter"
    iter_args = loop_pre[0].right.arguments

    assert len(iter_args) == 1
    assert isinstance(iter_args[0], Call)
    assert iter_args[0].func.name == "range"
    assert len(iter_args[0].arguments) == 3

    assert isinstance(iter_args[0].arguments[0], LiteralValue)
    assert iter_args[0].arguments[0].value == "1"
    assert isinstance(iter_args[0].arguments[1], LiteralValue)
    assert iter_args[0].arguments[1].value == "10"
    assert isinstance(iter_args[0].arguments[2], LiteralValue)
    assert iter_args[0].arguments[2].value == "1"

    assert isinstance(loop_pre[1], Assignment)
    assert isinstance(loop_pre[1].left, LiteralValue)
    assert loop_pre[1].left.value_type == "Tuple"

    assert isinstance(loop_pre[1].left.value[0], Var)
    assert loop_pre[1].left.value[0].val.name == "i"
    assert isinstance(loop_pre[1].left.value[1], Var)
    assert loop_pre[1].left.value[1].val.name == "generated_iter_0"
    assert isinstance(loop_pre[1].left.value[2], Var)
    assert loop_pre[1].left.value[2].val.name == "sc_0"
    
    assert isinstance(loop_pre[1].right, Call)
    assert loop_pre[1].right.func.name == "next"
    assert len(loop_pre[1].right.arguments) == 1
    assert loop_pre[1].right.arguments[0].val.name == "generated_iter_0"

    # Loop Test
    loop_test = loop_node.expr 
    assert isinstance(loop_test, Operator)
    assert loop_test.op == "ast.Eq"
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "sc_0"

    assert isinstance(loop_test.operands[1], LiteralValue)
    assert loop_test.operands[1].value_type == "Boolean"

    # Loop Body
    loop_body = loop_node.body
    next_call = loop_body[-1]
    assert isinstance(next_call, Assignment)
    assert isinstance(next_call.right, Call)
    assert next_call.right.func.name == "next"
    assert next_call.right.arguments[0].val.name == "generated_iter_0"


def test_for2():
    cast = generate_cast(for2())

    asg_node = cast.nodes[0].body[0]
    loop_node = cast.nodes[0].body[1]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '1'

    assert isinstance(loop_node, Loop)
    assert len(loop_node.pre) == 2

    # Loop Pre
    loop_pre = loop_node.pre
    assert isinstance(loop_pre[0], Assignment)
    assert isinstance(loop_pre[0].left, Var)
    assert loop_pre[0].left.val.name == "generated_iter_0"

    assert isinstance(loop_pre[0].right, Call)
    assert loop_pre[0].right.func.name == "iter"
    iter_args = loop_pre[0].right.arguments

    assert len(iter_args) == 1
    assert isinstance(iter_args[0], Call)
    assert iter_args[0].func.name == "range"
    assert len(iter_args[0].arguments) == 3

    assert isinstance(iter_args[0].arguments[0], LiteralValue)
    assert iter_args[0].arguments[0].value == "1"
    assert isinstance(iter_args[0].arguments[1], LiteralValue)
    assert iter_args[0].arguments[1].value == "10"
    assert isinstance(iter_args[0].arguments[2], LiteralValue)
    assert iter_args[0].arguments[2].value == "1"

    assert isinstance(loop_pre[1], Assignment)
    assert isinstance(loop_pre[1].left, LiteralValue)
    assert loop_pre[1].left.value_type == "Tuple"

    assert isinstance(loop_pre[1].left.value[0], LiteralValue)
    assert loop_pre[1].left.value[0].value_type == "Tuple"

    assert isinstance(loop_pre[1].left.value[0].value[0], Var)
    assert loop_pre[1].left.value[0].value[0].val.name == "a"
    assert isinstance(loop_pre[1].left.value[0].value[1], Var)
    assert loop_pre[1].left.value[0].value[1].val.name == "b"

    assert isinstance(loop_pre[1].left.value[1], Var)
    assert loop_pre[1].left.value[1].val.name == "generated_iter_0"
    assert isinstance(loop_pre[1].left.value[2], Var)
    assert loop_pre[1].left.value[2].val.name == "sc_0"
    
    assert isinstance(loop_pre[1].right, Call)
    assert loop_pre[1].right.func.name == "next"
    assert len(loop_pre[1].right.arguments) == 1
    assert loop_pre[1].right.arguments[0].val.name == "generated_iter_0"

    # Loop Test
    loop_test = loop_node.expr 
    assert isinstance(loop_test, Operator)
    assert loop_test.op == "ast.Eq"
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "sc_0"

    assert isinstance(loop_test.operands[1], LiteralValue)
    assert loop_test.operands[1].value_type == "Boolean"

    # Loop Body
    loop_body = loop_node.body
    body_asg = loop_body[0]
    assert isinstance(body_asg, Assignment)

    assert isinstance(body_asg.right, Operator)
    assert isinstance(body_asg.right.operands[0], Operator)
    assert isinstance(body_asg.right.operands[0].operands[0], Name)
    assert body_asg.right.operands[0].operands[0].name == "x"

    assert isinstance(body_asg.right.operands[0].operands[1], Name)
    assert body_asg.right.operands[0].operands[1].name == "a"

    assert isinstance(body_asg.right.operands[1], Name)
    assert body_asg.right.operands[1].name == "b"

    next_call = loop_body[-1]
    assert isinstance(next_call, Assignment)
    assert isinstance(next_call.right, Call)
    assert next_call.right.func.name == "next"
    assert next_call.right.arguments[0].val.name == "generated_iter_0"


def test_for3():
    cast = generate_cast(for3())

    asg_node = cast.nodes[0].body[0]
    list_node = cast.nodes[0].body[1]
    loop_node = cast.nodes[0].body[2]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, LiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '1'

    assert isinstance(loop_node, Loop)
    assert len(loop_node.pre) == 2

    assert isinstance(list_node, Assignment)
    assert isinstance(list_node.left, Var)
    assert list_node.left.val.name == "L"

    assert isinstance(list_node.right, LiteralValue)
    assert list_node.right.value_type == "List"

    # Loop Pre
    loop_pre = loop_node.pre
    assert isinstance(loop_pre[0], Assignment)
    assert isinstance(loop_pre[0].left, Var)
    assert loop_pre[0].left.val.name == "generated_iter_0"

    assert isinstance(loop_pre[0].right, Call)
    assert loop_pre[0].right.func.name == "iter"
    iter_args = loop_pre[0].right.arguments

    assert len(iter_args) == 1
    assert isinstance(iter_args[0], Var)
    assert iter_args[0].val.name == "L"

    assert isinstance(loop_pre[1], Assignment)
    assert isinstance(loop_pre[1].left, LiteralValue)
    assert loop_pre[1].left.value_type == "Tuple"

    assert isinstance(loop_pre[1].left.value[0], Var)
    assert loop_pre[1].left.value[0].val.name == "i"
    assert isinstance(loop_pre[1].left.value[1], Var)
    assert loop_pre[1].left.value[1].val.name == "generated_iter_0"
    assert isinstance(loop_pre[1].left.value[2], Var)
    assert loop_pre[1].left.value[2].val.name == "sc_0"
    
    assert isinstance(loop_pre[1].right, Call)
    assert loop_pre[1].right.func.name == "next"
    assert len(loop_pre[1].right.arguments) == 1
    assert loop_pre[1].right.arguments[0].val.name == "generated_iter_0"

    # Loop Test
    loop_test = loop_node.expr 
    assert isinstance(loop_test, Operator)
    assert loop_test.op == "ast.Eq"
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "sc_0"

    assert isinstance(loop_test.operands[1], LiteralValue)
    assert loop_test.operands[1].value_type == "Boolean"

    # Loop Body
    loop_body = loop_node.body
    next_call = loop_body[-1]

    assert isinstance(next_call, Assignment)
    assert isinstance(next_call.right, Call)
    assert next_call.right.func.name == "next"
    assert next_call.right.arguments[0].val.name == "generated_iter_0"
