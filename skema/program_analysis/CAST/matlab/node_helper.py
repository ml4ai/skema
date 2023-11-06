from typing import List, Dict
from skema.program_analysis.CAST2FN.model.cast import SourceRef

from tree_sitter import Node

CONTROL_CHARACTERS = [
    ",",
    "=",
    "==",
    "(",
    ")",
    "(/",
    "/)",
    ":",
    "::",
    "+",
    "-",
    "*",
    "**",
    "/",
    ">",
    "<",
    "<=",
    ">=",
    "only",
]

class NodeHelper():
    def __init__(self, source: str, source_file_name: str):
        self.source = source
        self.source_file_name = source_file_name


    def get_source_ref(self, node: Node) -> SourceRef:
        """Given a node and file name, return a CAST SourceRef object."""
        row_start, col_start = node.start_point
        row_end, col_end = node.end_point
        return SourceRef(self.source_file_name, col_start, col_end, row_start, row_end)


    def get_identifier(self, node: Node) -> str:
        """Given a node, return the identifier it represents. ie. The code between node.start_point and node.end_point"""
        line_num = 0
        column_num = 0
        in_identifier = False
        identifier = ""
        for i, char in enumerate(self.source):
            if line_num == node.start_point[0] and column_num == node.start_point[1]:
                in_identifier = True
            elif line_num == node.end_point[0] and column_num == node.end_point[1]:
                break

            if char == "\n":
                line_num += 1
                column_num = 0
            else:
                column_num += 1

            if in_identifier:
                identifier += char

        return identifier

def get_first_child_by_type(node: Node, type: str, recurse=False):
    """Takes in a node and a type string as inputs and returns the first child matching that type. Otherwise, return None
    When the recurse argument is set, it will also recursivly search children nodes as well.
    """
    for child in node.children:
        if child.type == type:
            return child

    if recurse:
        for child in node.children:
            out = get_first_child_by_type(child, type, True)
            if out:
                return out
    return None


def get_children_by_types(node: Node, types: List):
    """Takes in a node and a list of types as inputs and returns all children matching those types. Otherwise, return an empty list"""
    return [child for child in node.children if child.type in types]


def get_first_child_index(node, type: str):
    """Get the index of the first child of node with type type."""
    for i, child in enumerate(node.children):
        if child.type == type:
            return i


def remove_comments(node: Node):
    """Remove comment nodes from tree-sitter parse tree"""
    # NOTE: tree-sitter Node objects are read-only, so we have to be careful about how we remove comments
    # The below has been carefully designed to work around this restriction.
    to_remove = sorted([index for index,child in enumerate(node.children) if child.type == "comment"], reverse=True)
    for index in to_remove:
        del node.children[index]

    for i in range(len(node.children)):
        node.children[i] = remove_comments(node.children[i])

    return node


def get_last_child_index(node, type: str):
    """Get the index of the last child of node with type type."""
    last = None
    for i, child in enumerate(node.children):
        if child.type == type:
            last = child
    return last


def get_control_children(node: Node):
    return get_children_by_types(node, CONTROL_CHARACTERS)


def get_non_control_children(node: Node):
    children = []
    for child in node.children:
        if child.type not in CONTROL_CHARACTERS:
            children.append(child)

    return children
