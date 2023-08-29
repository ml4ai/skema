import json
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, List, Union
from tree_sitter import Language, Parser, Node, Tree

from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH

class MATLAB_TREE_BUILDER(object):
    def __init__(self):
        
        # Create the tree-sitter MATLAB parser
        self.parser = Parser()
        self.parser.set_language(
            Language(
                Path(Path(__file__).parent, INSTALLED_LANGUAGES_FILEPATH),
                "matlab"
            )
        )

    # create a syntax tree based on the matlab grammar and an input string
    def get_tree(self, source: str):
        tree = self.parser.parse(bytes(source, "utf8"))
        # prune empty nodes from syntax tree
        clean_tree = Tree
        clean_tree.root_node = self.clean_tree(
            tree.root_node 
        )
        return clean_tree

    # Remove empty child nodes
    def clean_tree(self, node: Node):
        for child in node.children:
            if child.type == '\n': # empty child
                node.children.remove(child)
        for child in node.children:
            self.clean_tree(child)
        return node

    # display the node tree in pretty format
    def traverse_tree(self, node: Node, indent = ''):
        for child in node.children:
            print(indent + 'node: ' + child.type)
            self.traverse_tree(child, indent + '  ')

