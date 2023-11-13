from skema.program_analysis.CAST.matlab.tests.utils import (
    assert_assignment,
    assert_call,
    assert_identifier,
    cast_nodes
)

# Test CAST from functions

def no_test_function():
    """ Test function """

    source = """
        function both = add_them(x, y)
            both = x + y
    end
         = false
    """

    nodes = cast_nodes(source)
    assert len(nodes) == 2

    # identifier
    assert_assignment(nodes[0], left = 'x', right = 'y')
    assert_assignment(nodes[1], left = 'r', right = 'x')

def test_function_call():
    """ Test function call """

    source = 'x = subplot(3, 5, 7)'

    """
    SYNTAX TREE:
    assignment
      identifier
      =
      function_call
        identifier
        (
        arguments
          number
          ,
          number
          ,
          number
        )
    ;
    """
    nodes = cast_nodes(source)
    assert len(nodes) == 1

    assert_assignment(nodes[0], left='x')
    assert_call(nodes[0].right, func = 'subplot', arguments = ['3', '5', '7'])


