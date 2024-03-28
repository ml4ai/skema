import pytest
from tempfile import TemporaryDirectory
from pathlib import Path 

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Var,
    Call,
    Name,
    CASTLiteralValue,
    ModelIf,
    Loop,
    Operator
)

def for1():
    return """
program for1
integer :: x = 7
integer :: i

do i=1, 10
x = x + 1
end do
end program for1
    """



def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]

def test_for1():
    cast = generate_cast(for1())

    asg_node = cast.nodes[0].body[0]
    loop_node = cast.nodes[0].body[2]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '7'

    assert isinstance(loop_node, Loop)
    print(loop_node)
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

    assert isinstance(iter_args[0].arguments[0], CASTLiteralValue)
    assert iter_args[0].arguments[0].value == "1"
    assert isinstance(iter_args[0].arguments[1], CASTLiteralValue)
    assert iter_args[0].arguments[1].value == "10"
    assert isinstance(iter_args[0].arguments[2], CASTLiteralValue)
    assert iter_args[0].arguments[2].value == "1"

    assert isinstance(loop_pre[1], Assignment)
    assert isinstance(loop_pre[1].left, CASTLiteralValue)
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
    assert loop_test.op == "!="
    assert isinstance(loop_test.operands[0], Name)
    assert loop_test.operands[0].name == "sc_0"

    assert isinstance(loop_test.operands[1], CASTLiteralValue)
    assert loop_test.operands[1].value_type == "Boolean"

    # Loop Body
    loop_body = loop_node.body
    next_call = loop_body[-1]
    assert isinstance(next_call, Assignment)
    assert isinstance(next_call.right, Call)
    assert next_call.right.func.name == "next"
    assert next_call.right.arguments[0].val.name == "generated_iter_0"

