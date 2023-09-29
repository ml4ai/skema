from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from pathlib import Path
import json
import sys

""" A fast way to run any file type through the Tree-sitter MATLAB parser"""

n_files = len(sys.argv)

if n_files == 1:
    print("Please enter one or more input filenames")
else:
    done = list()
    for i in range (1, n_files):
        filepath = Path(__file__).parent / sys.argv[i]
        print(f"\nParsing MATLAB tree for {str(filepath)}:")
        parser = MatlabToCast(filepath)
        cast = parser.out_cast
        print(f"\nNumber of CAST objects produced = {str(len(cast))}")

        print('\nCAST object:')
        for c in cast:
            jd = json.dumps(
                c.to_json_object(),
                sort_keys=True,
                indent=2,
            )
            print(jd)
        done.append(filepath)

    print("\nFiles processed:")
    for d in done:
        print(f"  {d}")
    print()
