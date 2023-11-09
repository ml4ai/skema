""" all the tokens from the Waterloo model syntax tree """

""" model syntax keywords """
# commented out words are not used for CAST parsing.
SYNTAX_KEYWORDS = [
    'arguments',
    'assignment',
    'binary_operator',
    'block',
    'boolean_operator',
    'break_statement',
#    'case', 
    'case_clause',
    'cell',
    'command',
    'command_argument',
    'command_name',
    'comment',
    'comparison_operator',
    'continue_statement',
#    'else',
    'else_clause',
#    'elseif',
    'elseif_clause',
    'end',
    'field_expression',
    'for',
    'for_statement',
#    'function', 
    'function_arguments',
    'function_call',
    'function_definition',
    'function_output',
    'identifier',
#    'if',
    'if_statement',
    'iterator',
    'lambda',
    'line_continuation',
    'matrix',
    'multioutput_variable',
    'not_operator',
    'number',
#    'otherwise', 
    'otherwise_clause',
    'parenthesis',
    'postfix_operator',
    'range',
    'row',
    'spread_operator',
    'string',
    'string_content',
#    'switch',
    'switch_statement',
    'unary_operator'
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
