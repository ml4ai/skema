# parser to be tested
from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from pathlib import Path
import os

TEST_DATA_PATH = Path(__file__).parent / "data"

def test_parse_matlab_files():
    """
    Tests whether each matlab file 
    produces a single CAST parse
    """

    for filename in os.listdir(TEST_DATA_PATH):
        if (filename.endswith(".m")):
            filepath = Path(TEST_DATA_PATH) / filename
            print(f"Testing parse tree for {str(filepath)}")
            parser = MatlabToCast(filepath)
            cast = parser.out_cast
            assert not len(cast) == 0  

if __name__ == "__main__":
    test_parse_matlab_files()

