import os
import sys
import json
import argparse

from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.CAST2FN.visitors.cast_to_agraph_visitor import (
    CASTToAGraphVisitor,
)

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST

from typing import Optional

def matlab_to_cast(
    path,
    agraph=False,
    std_out=False,
    rawjson=False,
    cast_obj=False,
) -> Optional[CAST]:
    """Create a CAST object from a MATLAB file and serialize it to JSON.

    Args:
        path: Path to the MATLAB .m source file
        agraph: If true, a PDF visualization of the graph is created.
        astprint: View the AST using the astpp module.
        std_out: If true, the CAST JSON is printed to stdout instead
                 of written to a file.
        rawjson: If true, the raw JSON contents are printed to stdout.
        cast_obj: If true, returns the CAST as an object instead of printing to
                stdout.

    Returns:
        If cast_obj is set to True, returns the CAST as an object. Else,
        returns None.
    """


    out_cast = TS2CAST(path).out_cast

    file_name = os.path.basename(path)
    if agraph:
        V = CASTToAGraphVisitor(out_cast)
        last_slash_idx = file_name.rfind("/")
        file_ending_idx = file_name.rfind(".")
        pdf_file_name = (
            f"{file_name[last_slash_idx + 1 : file_ending_idx]}.pdf"
        )
        V.to_pdf(pdf_file_name)

    # Then, print CAST as JSON
    if cast_obj:
        return out_cast
    else:
        if rawjson:
            print(
                json.dumps(
                    out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )
        else:
            if std_out:
                print(out_cast.to_json_str())
            else:
                out_name = file_name.split(".")[0]
                print("Writing CAST to " + out_name + "--CAST.json")
                out_handle = open(out_name + "--CAST.json", "w")
                out_handle.write(out_cast.to_json_str())

