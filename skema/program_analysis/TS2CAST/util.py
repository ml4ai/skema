from skema.program_analysis.CAST2FN.model.cast import AstNode, LiteralValue, SourceRef


def generate_dummy_source_refs(node: AstNode) -> None:
    """ Walks a tree of AstNodes replacing any null SourceRefs with a dummy value"""
    if isinstance(node, LiteralValue) and not node.source_code_data_type:
        node.source_code_data_type = ["Fortran", "Fotran95", "None"]
    if not node.source_refs:
        node.source_refs = [SourceRef("", -1, -1, -1, -1)]

    for attribute_str in node.attribute_map:
        attribute = getattr(node, attribute_str)
        if isinstance(attribute, AstNode):
            generate_dummy_source_refs(attribute)
        elif isinstance(attribute, list):
            for element in attribute:
                if isinstance(element, AstNode):
                    generate_dummy_source_refs(element)


def preprocess(source_code: str) -> str:
    """
    Preprocesses Fortran source code:
    1. Replaces the first occurrence of '|' with '&' if it is the first non-whitespace character in the line.
    2. Adds an additional '&' to the previous line
    """
    processed_lines = []
    for i, line in enumerate(source_code.splitlines()):
        if line.lstrip().startswith("|"):
            line = line.replace("|", "&", 1)
            processed_lines[-1] += "&"
        processed_lines.append(line)
    return "\n".join(processed_lines)
