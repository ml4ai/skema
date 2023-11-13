""" all the tokens from the Waterloo model syntax tree """

""" Waterloo model syntax keywords """
# commented out words are not used for CAST parsing.
SYNTAX_KEYWORDS = [
    # keywords in syntax but not used by CAST
    # 'case',
    # 'else',
    # 'elseif',
    # 'function',
    # 'if',
    # 'otherwise',
    # 'while',
    # 'switch',

    # keywords we currently support
    'assignment',
    'binary_operator',
    'block',
    'boolean_operator',
    'case_clause',
    'cell',
    'comment',
    'comparison_operator',
    'else_clause',
    'elseif_clause',
    'end',
    'identifier',
    'if_statement',
    'number',
    'otherwise_clause',
    'row',
    'string',
    'string_content',
    'switch_statement',
    'unary_operator',

    # keywords to be supported
    'arguments',
    'break_statement',
    'command',
    'command_argument',
    'command_name',
    'continue_statement',
    'field_expression',
    'for',
    'for_statement',
    'function_arguments',
    'function_call',
    'function_definition',
    'function_output',
    'iterator',
    'lambda',
    'line_continuation',
    'matrix',
    'multioutput_variable',
    'not_operator',
    'parenthesis',
    'postfix_operator',
    'range',
    'spread_operator',

    # keywords not in model or CAST that we should support
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
