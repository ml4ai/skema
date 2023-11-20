from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import Loop

# Test the for loop and others
def no_test_for_loop():
    """ Test the MATLAB for loop syntax elements"""
    source = """
        for n = 1:10
            x = do_something(n)
        end
    """
    nodes = cast(source)
    check(nodes[0], Loop())
