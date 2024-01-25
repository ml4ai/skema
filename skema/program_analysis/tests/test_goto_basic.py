# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import (
    GrometFNModuleCollection,
    FunctionType,
)
import ast
from skema.program_analysis.tests.utils_test import create_temp_file, delete_temp_file

from skema.program_analysis.CAST.fortran.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import SourceRef
from skema.program_analysis.CAST2FN import cast
from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline

def goto0():
    return """
    SUBROUTINE GoToExample
    INTEGER:: i

    i = 1
10  CONTINUE
    i = i + 1

    IF (i <= 5) GO TO 10

    END SUBROUTINE GoToExample
    """

def generate_gromet(test_file_string):
    # How do we generate CAST for Fortran from here?
    create_temp_file(test_file_string, "f95")

    ts2cast = TS2CAST("temp.f95")
    out_cast = ts2cast.out_cast[0]
    gromet = ann_cast_pipeline(out_cast, gromet=True, to_file=False, from_obj=True)

    delete_temp_file("f95")

    return gromet

def test_exp0():
    goto_gromet = generate_gromet(goto0())
    #####
    # Checking FN containing label
    label_fn = goto_gromet.fn_array[0]
    assert len(label_fn.bf) == 5
    assert label_fn.bf[1].function_type == FunctionType.LABEL
    assert label_fn.bf[1].name == "10"

    assert len(label_fn.pif) == 2
    assert label_fn.pif[0].box == 2
    assert label_fn.pif[1].box == 4
    
    assert len(label_fn.pof) == 4
    assert label_fn.pof[0].box == 1
    assert label_fn.pof[0].name == "i"
    
    assert label_fn.pof[1].box == 2
    assert label_fn.pof[1].name == "i"
    
    assert label_fn.pof[2].box == 4
    assert label_fn.pof[2].name == "i"

    assert label_fn.pof[3].box == 5

    assert len(label_fn.wff) == 2
    assert label_fn.wff[0].src == 1
    assert label_fn.wff[0].tgt == 1

    assert label_fn.wff[1].src == 2
    assert label_fn.wff[1].tgt == 2

    assert len(label_fn.wfc) == 1
    assert label_fn.wfc[0].src == 1
    assert label_fn.wfc[0].tgt == 3
    
    assert len(label_fn.bc) == 1
    assert label_fn.bc[0].condition == 5
    assert label_fn.bc[0].body_if == 6

    assert len(label_fn.pic) == 1
    assert label_fn.pic[0].box == 1
    assert label_fn.pic[0].name == "i"

    assert len(label_fn.poc) == 1
    assert label_fn.poc[0].box == 1
    assert label_fn.poc[0].name == "i"
    
    #####
    # Checking goto call
    goto_call_fn = goto_gromet.fn_array[5]
    assert len(goto_call_fn.opi) == 1
    assert goto_call_fn.opi[0].box == 1

    assert len(goto_call_fn.opo) == 1
    assert goto_call_fn.opo[0].box == 1
    assert goto_call_fn.opo[0].name == "i"

    assert len(goto_call_fn.wfopi) == 1
    assert goto_call_fn.wfopi[0].src == 1
    assert goto_call_fn.wfopi[0].tgt == 1

    assert len(goto_call_fn.wopio) == 1
    assert goto_call_fn.wopio[0].src == 1
    assert goto_call_fn.wopio[0].tgt == 1

    assert len(goto_call_fn.bf) == 1
    assert goto_call_fn.bf[0].function_type == FunctionType.GOTO
    assert goto_call_fn.bf[0].body == 7

    assert len(goto_call_fn.pif) == 1
    assert goto_call_fn.pif[0].box == 1

    #####
    # Checking basic label computation
    goto_expr_fn = goto_gromet.fn_array[6]
    assert len(goto_expr_fn.opi) == 1    
    assert goto_expr_fn.opi[0].box == 1

    assert len(goto_expr_fn.opo) == 2    
    assert goto_expr_fn.opo[0].box == 1
    assert goto_expr_fn.opo[0].name == "fn_idx"

    assert goto_expr_fn.opo[1].box == 1
    assert goto_expr_fn.opo[1].name == "label"
    
    # Checks the correct FN is grabbed in the label computation
    assert len(goto_expr_fn.bf) == 2    
    assert goto_expr_fn.bf[0].value == 1    
    assert goto_expr_fn.bf[1].value == "10"    

    assert len(goto_expr_fn.pof) == 2    
    assert goto_expr_fn.pof[0].box == 1
    assert goto_expr_fn.pof[1].box == 2
    
    assert len(goto_expr_fn.wfopo) == 2    
    assert goto_expr_fn.wfopo[0].src == 1
    assert goto_expr_fn.wfopo[0].tgt == 1

    assert goto_expr_fn.wfopo[1].src == 2
    assert goto_expr_fn.wfopo[1].tgt == 2
    
