import sys
from tree_sitter import Language, Parser
from tree_sitter import Tree
from pathlib import Path
from skema.program_analysis.CAST.matlab.tree_utils import TreeUtils
from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH
)



""" Get the Tree-sitter syntax tree for MATLAB files"""

if __name__ == "__main__":
    if len(sys.argv) > 1:

        # Create the Tree-sitter parser for the language name
        parser = Parser()
        parser.set_language(
            Language(INSTALLED_LANGUAGES_FILEPATH, "matlab")
        )

        tree_utils = TreeUtils()

        for i in range(1, len(sys.argv)):
            path = Path(sys.argv[i])
            print("\n\nINPUT:")
            print(path.name)
            source = path.read_text().strip()
            print("\nSOURCE:")
            print(source)

            print("\nSYNTAX TREE:")
            tree = parser.parse(bytes(source, "utf8"))
            # tree = tree_utils.clean_tree(tree)

            tree_utils.print_tree(tree)

    else:
        print("Please enter at least one input file")

