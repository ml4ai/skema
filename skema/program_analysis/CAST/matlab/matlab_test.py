import json
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Language, Parser, Node, Tree
from skema.program_analysis.CAST.matlab.matlab_tree_builder import MATLAB_TREE_BUILDER


class MATLAB_TEST(object):
    def __init__(self):
        
        # get a tree-sitter tree based on source input
        self.matlab_tree_builder = MATLAB_TREE_BUILDER()

    def test(self, title: str, source: str):

        print('\n\n' + title)
        print('\nSOURCE:')
        print(source)

        print('\nSYNTAX TREE:')
        tree: Tree = self.matlab_tree_builder.get_tree(source)
        self.matlab_tree_builder.traverse_nodes(tree.root_node)

        return tree


        
tester = MATLAB_TEST()

tree: Tree = tester.test(title = 'test 1', source = 'x = y * 5')

