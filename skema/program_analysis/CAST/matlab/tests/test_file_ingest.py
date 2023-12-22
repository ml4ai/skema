import os.path
from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import Assignment

def test_file_ingest():
    """ Test the ability of the CAST translator to read from a file"""

    filepath = "skema/program_analysis/CAST/matlab/tests/data/matlab.m"
    if not os.path.exists(filepath):
        filepath = "data/matlab.m"



    cast = MatlabToCast(source_path = filepath).out_cast
    module = cast.nodes[0]
    nodes = module.body
    check(nodes[0], Assignment(left = "y", right = "b"))
