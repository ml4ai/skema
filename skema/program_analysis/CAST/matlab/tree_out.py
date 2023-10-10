import sys
from tree_sitter import Language, Node, Parser
from pathlib import Path
from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH
)

""" Get the Tree-sitter syntax tree for MATLAB files"""

def print_tree(node: Node, indent = ''):
    """Display the node branch in pretty format"""
    for child in node.children:
        print(f"{indent} node: {child.type}")
        print_tree(child, indent + '  ')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Create the Tree-sitter parser for the language name
        parser = Parser()
        parser.set_language(
            Language(INSTALLED_LANGUAGES_FILEPATH, "matlab")
        )
        for i in range(1, len(sys.argv)):
            path = Path(sys.argv[i])
            print("\n\nINPUT:")
            print(path.name)
            source = path.read_text().strip()
            print("\nSOURCE:")
            print(source)
            print("\nSYNTAX TREE:")
            tree = parser.parse(bytes(source, "utf8"))
            print_tree(tree.root_node)
    else:
        print("Please enter at least one input file")
