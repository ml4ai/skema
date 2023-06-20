from typing import List
from skema.program_analysis.CAST2FN.model.cast import AstNode, LiteralValue, SourceRef


def generate_dummy_source_refs(node: AstNode) -> AstNode:
    """Walks a tree of AstNodes replacing any null SourceRefs with a dummy value"""
    if isinstance(node, LiteralValue) and not node.source_code_data_type:
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

