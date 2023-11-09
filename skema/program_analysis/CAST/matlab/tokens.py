""" all the tokens from the Waterloo model syntax tree """

class WaterlooTokens():

    """ syntax keywords """
    SYNTAX_KEYWORDS = [
        'arguments',
        'assignment',
        'binary_operator',
        'block',
        'boolean_operator',
        'break_statement',
        'case',
        'case_clause',
        'cell',
        'command',
        'command_argument',
        'command_name',
        'comment',
        'comparison_operator',
        'continue_statement',
        'else',
        'else_clause',
        'elseif',
        'elseif_clause',
        'end',
        'field_expression',
        'for',
        'for_statement',
        'function',
        'function_arguments',
        'function_call',
        'function_definition',
        'function_output',
        'identifier',
        'if',
        'if_statement',
        'iterator',
        'lambda',
        'line_continuation',
        'matrix',
        'multioutput_variable',
        'not_operator',
        'number',
        'otherwise',
        'otherwise_clause',
        'parenthesis',
        'postfix_operator',
        'range',
        'row',
        'spread_operator',
        'string',
        'string_content',
        'switch',
        'switch_statement',
        'unary_operator'
    ]

    """ Keywords used by CAST but not found in the Waterloo model """
    CAST_KEYWORDS = [
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

    """ the Waterloo model token zoo """
    NODE_TYPES = SYNTAX_KEYWORDS + CAST_KEYWORDS + OTHER_TOKENS
