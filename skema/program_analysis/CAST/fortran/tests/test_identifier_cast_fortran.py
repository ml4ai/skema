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

def identifier1():
    return """
        program test_identifier
        integer :: x = 2
        end program test_identifier
    """

def generate_cast(test_file_string):
    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "source.f95"
        source_path.write_text(test_file_string)
        out_cast = TS2CAST(str(source_path)).out_cast

    return out_cast[0]


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

