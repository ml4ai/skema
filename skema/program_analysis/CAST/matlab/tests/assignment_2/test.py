import os
import pytest

from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast

TEST_FILE = 'multi_line_assignment.m'

# test a single-line assignment
def test_single_assignment(tmpdir):
    parser = MatlabToCast(TEST_FILE)
    cast = parser.out_cast
    assert len(cast) == 1
