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
        # remove empty nodes from syntax tree
        return self.clean_tree(tree)

    # Remove empty children from the node tree
    def clean_nodes(self, node: Node):
        for child in node.children:
            if child.type == '\n': # empty child
                node.children.remove(child)
            else:
                self.clean_nodes(child)
        return node

    # display the node tree in pretty format
    def traverse_nodes(self, node: Node, indent = ''):
        for child in node.children:
            print(indent + 'node: ' + child.type)
            self.traverse_nodes(child, indent + '  ')

    # Clean the tree starting at the root node
    def clean_tree(self, tree:Tree):
        # prune empty nodes from syntax tree
        foo = Tree
        foo.root_node = self.clean_nodes(tree.root_node )
        return foo

    # Display the tree starting at the root node
    def traverse_tree(self, tree: Tree, indent = ''):
        self.traverse_nodes(tree.root_node, indent)
