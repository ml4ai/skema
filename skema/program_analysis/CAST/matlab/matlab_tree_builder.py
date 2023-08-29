import json
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Language, Parser, Node, Tree

from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH

#TODO:  Get from grammar
MATLAB_VERSION='matlab_version_here'

class MATLAB_PARSER(object):
    def __init__(self, source_file_path: str):
        
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
            tree.root_node, 
            clean_tree.root_node
        )
        return clean_tree

    # Remove empty nodes from tree
    def clean_tree(self, root: Tree, ret: Tree):
        for child in root.children:
            if child.type == '\n': # empty child
                root.children.remove(child)
        for child in root.children:
            self.clean_tree(child, root)
        return root

    # display the tree-sitter.TREE in pretty format
    def traverse_tree(self, root: Tree, indent):
        for child in root.children:
            print(indent + 'node: ' + child.type)
            self.traverse_tree(child, indent + '  ')
