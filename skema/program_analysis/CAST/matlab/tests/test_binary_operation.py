from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from pathlib import Path

# Test the CAST returned by processing the simplest MATLAB binary assignment

# The source code for the single line assignment test is 'z = x + y;'
#  SYNTAX TREE:
#  node: assignment
#    node: identifier 'z'
#    node: =
#    node: binary_operator
#      node: identifier 'x'
#      node: +
#      node: identifier 'y'

def test_binary_operation():
    """ Tests whether the parser can handle a binary operation """
    filepath = Path(__file__).parent / "data" / "binary_assignment.m"
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
    assert module['left']['val']['name'] == 'z'
    assert module['right']['node_type'] == 'Operator'
    assert module['right']['op'] == '+'
    assert module['right']['operands'][0]['node_type'] == 'Var'
    assert module['right']['operands'][0]['val']['name'] == 'x'
    assert module['right']['operands'][1]['node_type'] == 'Var'
    assert module['right']['operands'][1]['val']['name'] == 'y'
    # ...

if __name__ == "__main__":
    test_binary_operation()
