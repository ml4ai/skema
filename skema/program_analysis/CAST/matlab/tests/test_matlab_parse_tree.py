import requests
import os
from pathlib import Path

from tree_sitter import Language, Parser

# test parser
from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast

# test for existence of shared grammar object
from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH
if INSTALLED_LANGUAGES_FILEPATH.exists()
    print('Shared object at: ' + str(INSTALLED_LANGUAGES_FILEPATH))
else:  
    # If not found, create it
    print('Did not find shared object at ' + str(INSTALLED_LANGUAGES_FILEPATH))
    from skema.program_analysis.tree_sitter_parsers.build_parsers import build_parsers
    build_parsers(["matlab"])
    

TEST_DATA_PATH = 'data/'

for filename in os.listdir(TEST_DATA_PATH):
    if (filename.endswith('.m')):
        parser = MatlabToCast(TEST_DATA_PATH + filename)

        cast = parser.out_cast
        assert len(cast) == 1

