from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from pathlib import Path

# Test the CAST returned by processing a single line MATLAB source file

# The source code for the single line assignment test is 'x = 5'
# The correct CAST for this single line assignment would be a single node: 
#   node_type 'Assignment' 
#   left.val.name 'x'
#   right.value '5'
#   right.value_type 'Integer'

def test_single_assignment():
    """ Tests whether the parser can handle a single assignment statement """
    filepath = Path(__file__).parent / "data" / "single_line_assignment.m"
    print(f"Testing parse tree for {str(filepath)}")
    parser = MatlabToCast(filepath)
    cast = parser.out_cast

    # there should only be one CAST object in the cast output list
    assert len(cast) == 1  

    head = cast[0].to_json_object()
    nodes = head['nodes']

    # there should be only one node in this single-line assignment test
    assert len(nodes) == 1

    body = nodes[0]['body']

    # There should only be one module within the node body 
    assert len(body) == 1

    # Test the module dictionary for relevant key values
    module = body[0]
    assert module['node_type'] == 'Assignment'
    assert module['left']['val']['name'] == 'x'
    assert module['right']['value_type'] == 'Integer'
    assert module['right']['value'] == '5'
