from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    first_cast_node
)

# Test the CAST returned by processing the simplest MATLAB assignment

def test_assignment():
    """ Test CAST from MATLAB 'assignment' statement."""

    source = 'x = 5'
    
    # The root of the CAST should be Assignment
    assignment = first_cast_node(source)

    # The module body should contain a single assignment node
    assert_assignment(assignment, left = "x", right = "5")
