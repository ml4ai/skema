import json
import sys
from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast

""" Run a file of any type through the Tree-sitter MATLAB parser"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            parser = MatlabToCast(sys.argv[i])
            print("\n\nINPUT:")
            print(parser.filename)
            print("\nSOURCE:")
            print(parser.source)
            print('\nCAST:')
            cast_list = parser.out_cast
            for cast_index in range(0, len(cast_list)):
                jd = json.dumps(
                    cast_list[cast_index].to_json_object(),
                    sort_keys=True,
                    indent=2,
                )
                print(jd)
    else:
        print("Please enter one filename to parse")
