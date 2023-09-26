import requests
import os

from tree_sitter import Language, Parser

# test parser
from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast

TEST_DATA_PATH = 'data'

def test_parse_matlab_files():
    """
    Tests whether each matlab file 
    produces a single CAST parse
    """
    for filename in os.listdir(TEST_DATA_PATH):
        if (filename.endswith('.m')):
            filepath = os.path.join(TEST_DATA_PATH, filename)
            print('Testing parse tree for ' + str(filepath))
            parser = MatlabToCast(filepath)
            cast = parser.out_cast
            assert len(cast) == 1  


test_parse_matlab_files()
