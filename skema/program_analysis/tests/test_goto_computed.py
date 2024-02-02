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
PROGRAM ComputedGoTo
      INTEGER:: a, b, c

      a = 1
      b = 2
      c = 3

      GO TO (100, 200, 300, 400), MOD((a + b) * c / 2, 4) + 1

  100 PRINT *, 'Resulting choice led to label 100'
      GO TO 500

  200 PRINT *, 'Resulting choice led to label 200'
      GO TO 500

  300 PRINT *, 'Resulting choice led to label 300'
      GO TO 500

  400 PRINT *, 'Resulting choice led to label 400'
      GO TO 500

  500 PRINT *, 'End of program'
      END PROGRAM ComputedGoTo    
    """

def goto1():
    return """
      SUBROUTINE EXAMPLE(DA,C,S)
*     .. Scalar Arguments ..
      DOUBLE PRECISION C,DA,S
*     .. Local Scalars ..
      DOUBLE PRECISION R,ROE
      IF (DA.NE.10) GO TO 10
      C = 1
      R = 2
      GO TO 20
   10 C = 2
      R = 3
   20 DA = R
      S = C
      RETURN
      END
    """
    

def generate_gromet(test_file_string):
    # How do we generate CAST for Fortran from here?
    create_temp_file(test_file_string, "f95")

    ts2cast = TS2CAST("temp.f95")
    out_cast = ts2cast.out_cast[0]
    gromet = ann_cast_pipeline(out_cast, gromet=True, to_file=False, from_obj=True)

    delete_temp_file("f95")

    return gromet

def test_goto0():
    goto_gromet = generate_gromet(goto0())
    #####
    # Checking FN containing label
    base_fn = goto_gromet.fn
    assert base_fn.bf[3].function_type == FunctionType.GOTO
    assert base_fn.bf[6].function_type == FunctionType.GOTO
    assert base_fn.bf[9].function_type == FunctionType.GOTO
    assert base_fn.bf[12].function_type == FunctionType.GOTO
    assert base_fn.bf[15].function_type == FunctionType.GOTO

    assert base_fn.bf[4].function_type == FunctionType.LABEL
    assert base_fn.bf[7].function_type == FunctionType.LABEL
    assert base_fn.bf[10].function_type == FunctionType.LABEL
    assert base_fn.bf[13].function_type == FunctionType.LABEL
    assert base_fn.bf[16].function_type == FunctionType.LABEL

    assert base_fn.pif[0].box == 4
    assert base_fn.pif[1].box == 4
    assert base_fn.pif[2].box == 4

    assert base_fn.pif[3].box == 5
    assert base_fn.pif[4].box == 5
    assert base_fn.pif[5].box == 5

    assert base_fn.pif[6].box == 7
    assert base_fn.pif[7].box == 7
    assert base_fn.pif[8].box == 7

    assert base_fn.pif[9].box == 8
    assert base_fn.pif[10].box == 8
    assert base_fn.pif[11].box == 8

    assert base_fn.pif[12].box == 10
    assert base_fn.pif[13].box == 10
    assert base_fn.pif[14].box == 10

    assert base_fn.pif[15].box == 11
    assert base_fn.pif[16].box == 11
    assert base_fn.pif[17].box == 11

    assert base_fn.pif[18].box == 13
    assert base_fn.pif[19].box == 13
    assert base_fn.pif[20].box == 13

    assert base_fn.pif[21].box == 14
    assert base_fn.pif[22].box == 14
    assert base_fn.pif[23].box == 14

    assert base_fn.pif[24].box == 16
    assert base_fn.pif[25].box == 16
    assert base_fn.pif[26].box == 16

    assert base_fn.pif[27].box == 17
    assert base_fn.pif[28].box == 17
    assert base_fn.pif[29].box == 17

    assert base_fn.pof[0].box == 1
    assert base_fn.pof[1].box == 2
    assert base_fn.pof[2].box == 3

    assert base_fn.pof[3].box == 5
    assert base_fn.pof[4].box == 5
    assert base_fn.pof[5].box == 5

    assert base_fn.pof[6].box == 8
    assert base_fn.pof[7].box == 8
    assert base_fn.pof[8].box == 8

    assert base_fn.pof[9].box == 11
    assert base_fn.pof[10].box == 11
    assert base_fn.pof[11].box == 11

    assert base_fn.pof[12].box == 14
    assert base_fn.pof[13].box == 14
    assert base_fn.pof[14].box == 14

    assert base_fn.pof[15].box == 17
    assert base_fn.pof[16].box == 17
    assert base_fn.pof[17].box == 17


    # First Goto computation with expression
    #########
    goto_expr_fn = goto_gromet.fn_array[3]
    assert len(goto_expr_fn.bf) == 2
    assert goto_expr_fn.bf[0].function_type == FunctionType.EXPRESSION
    assert goto_expr_fn.bf[0].body == 6

    assert goto_expr_fn.bf[1].function_type == FunctionType.LITERAL
    assert goto_expr_fn.bf[1].value.value_type == "None"
    assert goto_expr_fn.bf[1].value.value == "None"

    assert len(goto_expr_fn.opi) == 3
    assert goto_expr_fn.opi[0].box == 1
    assert goto_expr_fn.opi[1].box == 1
    assert goto_expr_fn.opi[2].box == 1

    assert len(goto_expr_fn.opo) == 2
    assert goto_expr_fn.opo[0].box == 1
    assert goto_expr_fn.opo[0].name == "fn_idx"

    assert goto_expr_fn.opo[1].box == 1
    assert goto_expr_fn.opo[1].name == "label"

    assert len(goto_expr_fn.pif) == 3
    assert goto_expr_fn.pif[0].box == 1
    assert goto_expr_fn.pif[1].box == 1
    assert goto_expr_fn.pif[2].box == 1
    
    assert len(goto_expr_fn.pof) == 2
    assert goto_expr_fn.pof[0].box == 1
    assert goto_expr_fn.pof[1].box == 2

    assert len(goto_expr_fn.wfopi) == 3
    assert goto_expr_fn.wfopi[0].src == 1
    assert goto_expr_fn.wfopi[0].tgt == 1

    assert goto_expr_fn.wfopi[1].src == 2
    assert goto_expr_fn.wfopi[1].tgt == 2

    assert goto_expr_fn.wfopi[2].src == 3
    assert goto_expr_fn.wfopi[2].tgt == 3

    assert len(goto_expr_fn.wfopo) == 2
    assert goto_expr_fn.wfopo[0].src == 1
    assert goto_expr_fn.wfopo[0].tgt == 1

    assert goto_expr_fn.wfopo[1].src == 2
    assert goto_expr_fn.wfopo[1].tgt == 2
    
    #####
    # Checking basic label computation
    # Multiples of these exist in the FN but they're identical
    goto_expr_fn = goto_gromet.fn_array[7]
    assert len(goto_expr_fn.opi) == 3    
    assert goto_expr_fn.opi[0].box == 1
    assert goto_expr_fn.opi[1].box == 1
    assert goto_expr_fn.opi[2].box == 1

    assert len(goto_expr_fn.opo) == 2    
    assert goto_expr_fn.opo[0].box == 1
    assert goto_expr_fn.opo[0].name == "fn_idx"

    assert goto_expr_fn.opo[1].box == 1
    assert goto_expr_fn.opo[1].name == "label"
    
    # Checks the correct FN is grabbed in the label computation
    assert len(goto_expr_fn.bf) == 2    
    assert goto_expr_fn.bf[0].value == 0    
    assert goto_expr_fn.bf[1].value == "500"    

    assert len(goto_expr_fn.pof) == 2    
    assert goto_expr_fn.pof[0].box == 1
    assert goto_expr_fn.pof[1].box == 2
    
    assert len(goto_expr_fn.wfopo) == 2    
    assert goto_expr_fn.wfopo[0].src == 1
    assert goto_expr_fn.wfopo[0].tgt == 1

    assert goto_expr_fn.wfopo[1].src == 2
    assert goto_expr_fn.wfopo[1].tgt == 2

    #####
    # Checking computed label computation
    goto_expr_fn = goto_gromet.fn_array[5]
    assert len(goto_expr_fn.opi) == 3    
    assert goto_expr_fn.opi[0].box == 1
    assert goto_expr_fn.opi[1].box == 1
    assert goto_expr_fn.opi[2].box == 1

    assert len(goto_expr_fn.opo) == 1    
    assert goto_expr_fn.opo[0].box == 1
    
    # Checks the label generation for the computed GOTO is correct
    assert len(goto_expr_fn.bf) == 10
    assert goto_expr_fn.bf[0].function_type == FunctionType.LANGUAGE_PRIMITIVE
    assert goto_expr_fn.bf[0].name == "_get"

    assert goto_expr_fn.bf[2].function_type == FunctionType.IMPORTED_METHOD
    assert goto_expr_fn.bf[2].body == 5

    assert len(goto_expr_fn.pif) == 12    
    assert goto_expr_fn.pif[0].box == 1
    assert goto_expr_fn.pif[1].box == 4
    assert goto_expr_fn.pif[2].box == 4
    assert goto_expr_fn.pif[3].box == 5
    assert goto_expr_fn.pif[4].box == 5
    assert goto_expr_fn.pif[5].box == 7
    assert goto_expr_fn.pif[6].box == 7
    assert goto_expr_fn.pif[7].box == 3
    assert goto_expr_fn.pif[8].box == 3
    assert goto_expr_fn.pif[9].box == 10
    assert goto_expr_fn.pif[10].box == 10
    assert goto_expr_fn.pif[11].box == 1

    assert len(goto_expr_fn.pof) == 10    
    assert goto_expr_fn.pof[0].box == 1
    assert goto_expr_fn.pof[1].box == 2
    assert goto_expr_fn.pof[2].box == 4
    assert goto_expr_fn.pof[3].box == 5
    assert goto_expr_fn.pof[4].box == 6
    assert goto_expr_fn.pof[5].box == 7
    assert goto_expr_fn.pof[6].box == 8
    assert goto_expr_fn.pof[7].box == 3
    assert goto_expr_fn.pof[8].box == 9
    assert goto_expr_fn.pof[9].box == 10

    assert len(goto_expr_fn.wfopi) == 3    
    assert goto_expr_fn.wfopi[0].src == 2
    assert goto_expr_fn.wfopi[0].tgt == 1

    assert goto_expr_fn.wfopi[1].src == 3
    assert goto_expr_fn.wfopi[1].tgt == 2

    assert goto_expr_fn.wfopi[2].src == 5
    assert goto_expr_fn.wfopi[2].tgt == 3

    assert len(goto_expr_fn.wff) == 9    
    assert goto_expr_fn.wff[0].src == 1
    assert goto_expr_fn.wff[0].tgt == 2
    
    assert goto_expr_fn.wff[1].src == 4
    assert goto_expr_fn.wff[1].tgt == 3

    assert goto_expr_fn.wff[2].src == 6
    assert goto_expr_fn.wff[2].tgt == 4
    
    assert goto_expr_fn.wff[3].src == 7
    assert goto_expr_fn.wff[3].tgt == 5

    assert goto_expr_fn.wff[4].src == 8
    assert goto_expr_fn.wff[4].tgt == 6

    assert goto_expr_fn.wff[5].src == 9
    assert goto_expr_fn.wff[5].tgt == 7

    assert goto_expr_fn.wff[6].src == 10
    assert goto_expr_fn.wff[6].tgt == 8

    assert goto_expr_fn.wff[7].src == 11
    assert goto_expr_fn.wff[7].tgt == 9

    assert goto_expr_fn.wff[8].src == 12
    assert goto_expr_fn.wff[8].tgt == 10
    
    assert len(goto_expr_fn.wfopo) == 1    
    assert goto_expr_fn.wfopo[0].src == 1
    assert goto_expr_fn.wfopo[0].tgt == 1