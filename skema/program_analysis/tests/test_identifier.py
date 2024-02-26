# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
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

def identifier1():
    return """x = 2"""

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast


# Tests to make sure that identifiers are correctly being generated
def test_identifier1():
    cast = generate_cast(identifier1())

    asg_node = cast.nodes[0].body[0]

    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert isinstance(asg_node.left.val, Name)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, CASTLiteralValue)
    assert asg_node.right.value_type == "Integer"
    assert asg_node.right.value == '2'

