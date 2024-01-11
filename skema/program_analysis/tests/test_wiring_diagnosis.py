from skema.program_analysis.gromet_wire_diagnosis import check_wire
from skema.gromet.fn import GrometWire


def test_correct_wire():
    correct_wire = GrometWire(src=1, tgt=1)
    result = check_wire(correct_wire, 1, 1, "wff")
    assert not result

    correct_wire = GrometWire(src=3, tgt=4)
    result = check_wire(correct_wire, 4, 5, "wlc")
    assert not result

    correct_wire = GrometWire(src=2, tgt=1)
    result = check_wire(correct_wire, 2, 1, "wff")

def test_wrong_wire():
    wrong_wire = GrometWire(src=0, tgt=-1)
    result = check_wire(wrong_wire, 1, 1, "wff")
    assert result

    wrong_wire = GrometWire(src=20, tgt=2)
    result = check_wire(wrong_wire, 19, 2, "wff")
    assert result

    wrong_wire = GrometWire(src=-1, tgt=2)
    result = check_wire(wrong_wire, 1, 1, "wlc")
    assert result
