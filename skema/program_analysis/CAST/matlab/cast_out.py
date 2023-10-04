from skema.program_analysis.CAST.matlab.matlab_to_cast import MatlabToCast
from skema.program_analysis.CAST.matlab.tree_builder import TreeBuilder
from tree_sitter import Tree

import json
import sys

def show_source(parser):
    """ Show the input source file """
    print('\nSOURCE:')
    print(parser.source)

def show_cast(parser):
    """ Show the CAST structures """
    print('\nCAST:')
    cast_list = parser.out_cast
    for cast_index in range(0, len(cast_list)):
        jd = json.dumps(
            cast_list[cast_index].to_json_object(),
            sort_keys=True,
            indent=2,
        )
        print(jd)

def show_tree(parser):
    """ Show the Tree-sitter syntax tree """
    print('\nSYNTAX TREE:')
    tree_builder = parser.tree_builder
    tree_builder.print_tree(parser.tree)


""" Run a file of any type through the Tree-sitter MATLAB parser"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            parser = MatlabToCast(sys.argv[i])
            show_source(parser)
            show_cast(parser)
            show_tree(parser)

    else:
        print("Please enter one filename to parse")

