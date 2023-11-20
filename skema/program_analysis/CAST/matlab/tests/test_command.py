from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (Call)

def test_command():
    """ Test the MATLAB command syntax elements"""

    nodes = cast("clear all;")
    check(nodes[0], Call(func = "clear", arguments=["all"]))
