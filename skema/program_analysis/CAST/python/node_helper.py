import itertools
from typing import List, Dict
from skema.program_analysis.CAST2FN.model.cast import SourceRef

from tree_sitter import Node

CONTROL_CHARACTERS = [
    ",",
    "=",
    "==",
    "(",
    ")",
    ":",
    "+",
    "-",
    "*",
    "**",
    "/",
    "!="
    ">",
    "<",
    "<=",
    ">=",
    "in",
    "not"
]

# Whatever constructs we see in the left
# part of the for loop construct
# for LEFT in RIGHT: 
FOR_LOOP_LEFT_TYPES = [
    "identifier",
    "tuple_pattern",
    "pattern_list",
    "list_pattern"
]

# Whatever constructs we see in the right
# part of the for loop construct
# for LEFT in RIGHT: 
FOR_LOOP_RIGHT_TYPES = [
    "call",
    "identifier",
    "list",
    "tuple"
]

# Whatever constructs we see in the conditional
# part of the while loop
WHILE_COND_TYPES = [
    "boolean_operator",
    "call",
    "comparison_operator"
]

class NodeHelper():
    def __init__(self, source: str, source_file_name: str):
        self.source = source
        self.source_file_name = source_file_name

        # get_identifier optimization variables
        self.source_lines = source.splitlines(keepends=True)
        self.line_lengths = [len(line) for line in self.source_lines]
        self.line_length_sums = [0] + list(itertools.accumulate(self.line_lengths))
    
    def get_identifier(self, node: Node) -> str:
        """Given a node, return the identifier it represents. ie. The code between node.start_point and node.end_point"""
        start_line, start_column = node.start_point
        end_line, end_column = node.end_point

        # Edge case for when an identifier is on the very first line of the code
        # We can't index into the line_length_sums
        start_index = self.line_length_sums[start_line] + start_column        
        if start_line == end_line:
            end_index = start_index + (end_column-start_column)
        else:
            end_index = self.line_length_sums[end_line] + end_column

        return self.source[start_index:end_index]

    def get_source_ref(self, node: Node) -> SourceRef:
        """Given a node and file name, return a CAST SourceRef object."""
        row_start, col_start = node.start_point
        row_end, col_end = node.end_point
        return SourceRef(self.source_file_name, col_start, col_end, row_start, row_end)

    def get_operator(self, node: Node) -> str:
        """Given a unary/binary operator node, return the operator it contains"""
        return node.type

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

