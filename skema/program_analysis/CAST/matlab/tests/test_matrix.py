from skema.program_analysis.CAST.matlab.tests.utils import (check, cast)
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    LiteralValue
)

# Test CAST using matrices
def test_empty():
    """ Test empty matrix."""
    check(cast("[];")[0], [])

def test_booleans():
    """ Test matrix with MATLAB booleans."""
    check(cast("[true false];")[0], ["True", "False"])

def test_values():
    """ Test assignment 1 dimensional matrix value."""
    check(cast("[1 x 'Bob' ]")[0], [1, 'x', "'Bob'"])

