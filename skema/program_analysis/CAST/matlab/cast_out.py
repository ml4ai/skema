from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
import json
import sys

def cast_out(filename):
    """ Parse the file with the MATLAB parser"""
    parser = MatlabToCast(filename)
    cast_list = parser.out_cast
    for cast_index in range(0, len(cast_list)):
        jd = json.dumps(
            cast_list[cast_index].to_json_object(),
            sort_keys=True,
            indent=2,
        )
        print(jd)


""" Run a file of any type through the Tree-sitter MATLAB parser"""
if __name__ == "__main__":
    if len(sys.argv) == 2:
        cast_out(sys.argv[1])
    else:
        print("Please enter one filename to parse")

