from typing import List, Dict
from skema.program_analysis.CAST2FN.model.cast import SourceRef


class NodeHelper(object):
    def __init__(self, source_file_name: str, source: str):
        self.source_file_name = source_file_name
        self.source = source

    def parse_tree_to_dict(self, node) -> Dict:
        node_dict = {
            "type": self.get_node_type(node),
            "source_refs": [self.get_node_source_ref(node)],
            "identifier": self.get_node_identifier(node),
            "original_children_order": [],
            "children": [],
            "comments": [],
            "control": [],
        }

        for child in node.children:
            child_dict = self.parse_tree_to_dict(child)
            node_dict["original_children_order"].append(child_dict)
            if self.is_comment_node(child):
                node_dict["comments"].append(child_dict)
            elif self.is_control_character_node(child):
                node_dict["control"].append(child_dict)
            else:
                node_dict["children"].append(child_dict)

        return node_dict

    def is_comment_node(self, node):
        if node.type == "comment":
            return True
        return False

    def is_control_character_node(self, node):
        control_characters = [
            ",",
            "=",
            "(",
            ")",
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
        ]
        return node.type in control_characters

    def get_node_source_ref(self, node) -> SourceRef:
        row_start, col_start = node.start_point
        row_end, col_end = node.end_point
        return SourceRef(self.source_file_name, col_start, col_end, row_start, row_end)

    def get_node_identifier(self, node) -> str:
        source_ref = self.get_node_source_ref(node)

        line_num = 0
        column_num = 0
        in_identifier = False
        identifier = ""
        for i, char in enumerate(self.source):
            if line_num == source_ref.row_start and column_num == source_ref.col_start:
                in_identifier = True
            elif line_num == source_ref.row_end and column_num == source_ref.col_end:
                break

            if char == "\n":
                line_num += 1
                column_num = 0
            else:
                column_num += 1

            if in_identifier:
                identifier += char

        return identifier

    def get_node_type(self, node) -> str:
        return node.type

    def get_first_child_by_type(self, node: Dict, node_type: str) -> Dict:
        children = self.get_children_by_type(node, node_type)
        if len(children) >= 1:
            return children[0]

    def get_children_by_type(self, node: Dict, node_type: str) -> List:
        children = []

        for child in node["children"]:
            if child["type"] == node_type:
                children.append(child)

        return children
