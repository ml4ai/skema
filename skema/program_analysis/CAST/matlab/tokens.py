""" all the tokens from the Waterloo model syntax tree """

""" Waterloo model syntax keywords """
KEYWORDS = [
    # keywords we currently support
    'arguments',
    'assignment',
    'binary_operator',
    'block',
    'boolean',
    'boolean_operator',
    'case',
    'case_clause',
    'cell',
    'command',
    'command_argument',
    'command_name',
    'comment',
    'comparison_operator',
    'else',
    'else_clause',
    'elseif',
    'elseif_clause',
    'end',
    'function',
    'function_arguments',
    'function_call',
    'function_definition',
    'function_output',
    'identifier',
    'if',
    'if_statement',
    'matrix',
    'module'
    'not_operator',
    'number',
    'otherwise',
    'otherwise_clause',
    'parenthesis',
    'postfix_operator',
    'row',
    'source_file',
    'spread_operator',
    'string',
    'string_content',
    'switch',
    'switch_statement',
    'unary_operator',

    # keywords currently being added
    'break_statement',
    'continue_statement',
    'for',
    'for_statement',
    'iterator',
    'range',

    # keywords to be supported
    'field_expression',
    'lambda',
    'line_continuation',
    'multioutput_variable',
]

""" anything not a keyword """
OTHER_TOKENS = [
    '\"',
    '&',
    '&&',
    '\'',
    '(',
    ')',
    '*',
    '+',
    ',',
    '-',
    '.',
    '.*',
    './',
    '/',
    ':',
    ';',
    '<',
    '<=',
    '=',
    '==',
    '>',
    '>=',
    '@',
    '[',
    ']',
    '{',
    '||',
    '}',
    '~'
]

""" all tokens """
TOKENS = KEYWORDS + OTHER_TOKENS
