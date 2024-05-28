from typing import List
from skema.program_analysis.CAST2FN.model.cast import AstNode, CASTLiteralValue, SourceRef


def generate_dummy_source_refs(node: AstNode) -> AstNode:
    """Walks a tree of AstNodes replacing any null SourceRefs with a dummy value"""
    if isinstance(node, CASTLiteralValue) and not node.source_code_data_type:
        node.source_code_data_type = ["Fortran", "Fotran95", "None"]
    if not node.source_refs:
        node.source_refs = [SourceRef("", -1, -1, -1, -1)]

    for attribute_str in node.attribute_map:
        attribute = getattr(node, attribute_str)
        if isinstance(attribute, AstNode):
            generate_dummy_source_refs(attribute)
        elif isinstance(attribute, List):
            for element in attribute:
                if isinstance(element, AstNode):
                    generate_dummy_source_refs(element)

    return node

def get_op(operator):
    ops = {
        '+': 'ast.Add',
        '-': 'ast.Sub',
        '*': 'ast.Mult',
        '/': 'ast.Div',
        '==' : 'ast.Eq',
        '!=' : 'ast.NotEq',
        '<'  : 'ast.Lt',
        '<=' : 'ast.LtE',
        '>'  : 'ast.Gt',
        '>=' : 'ast.GtE',
        '%' : 'ast.Mod',
        'and' : 'ast.And',
        'or' : 'ast.Or',
        '**' : 'ast.Pow',
        'is' : 'ast.Is',
        'is not': 'ast.IsNot',
        '>>' : 'ast.RShift',
        '<<' : 'ast.LShift',
        '|' : 'ast.BitOr',
        '&' : 'ast.BitAnd',
        '~' : 'ast.Invert',
        '^' : 'ast.BitXor',
        '//' : 'ast.FloorDiv'
        # ast.UAdd: 'ast.UAdd',
        # ast.USub: 'ast.USub',
        # ast.And: 'ast.And',
        # ast.Or: 'ast.Or',
        # ast.In: 'ast.In',
        # ast.NotIn: 'ast.NotIn',
        # ast.Not: 'ast.Not',
    }
    if operator in ops.keys():
        return ops[operator]
    else:
        raise NotImplementedError(f"python tree sitter util.py: Operator {operator} isn't in the operations table")
