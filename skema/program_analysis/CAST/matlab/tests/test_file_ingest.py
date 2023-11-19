from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
import os

from skema.program_analysis.CAST2FN.model.cast import Assignment

def test_file_ingest():
    """ Test the ability of the CAST translator to read from a file"""

    filename = "skema/program_analysis/CAST/matlab/tests/data/matlab.m"
    cast = MatlabToCast(source_path = filename).out_cast
    module = cast.nodes[0]
    nodes = module.body
    check(nodes[0], Assignment(left = "y", right = "b"))
