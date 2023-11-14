""" all the tokens from the Waterloo model syntax tree """

""" Waterloo model syntax keywords """
# commented out words are not used for CAST parsing.
SYNTAX_KEYWORDS = [
    # keywords we currently support
    'assignment',
    'binary_operator',
    'block',
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
    'function_call',
    'identifier',
    'if',
    'if_statement',
    'matrix',
    'number',
    'otherwise',
    'otherwise_clause',
    'row',
    'string',
    'string_content',
    'switch',
    'switch_statement',
    'unary_operator',

    # keywords to be supported
    'arguments',
    'break_statement',
    'continue_statement',
    'field_expression',
    'for',
    'for_statement',
    'function_arguments',
    'function_definition',
    'function_output',
    'iterator',
    'lambda',
    'line_continuation',
    'multioutput_variable',
    'not_operator',
    'parenthesis',
    'postfix_operator',
    'range',
    'spread_operator',
    'while',
    'while_statement' 
]

""" Keywords used by CAST but not found in the Waterloo model """
OTHER_KEYWORDS = [
    'import',
    'source_file',
    'module'
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

""" all keywords """
KEYWORDS = SYNTAX_KEYWORDS + OTHER_KEYWORDS

""" all tokens """
TOKENS = KEYWORDS + OTHER_TOKENS
