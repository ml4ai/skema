from tree_sitter import Language, Parser, Node, Tree
from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH
)

# Create a Tree-sitter grammar tree using the language name and an input file

class TreeBuilder(object):
    def __init__(self, language_name):
        """Build a parser for the language name, ie 'fortran' """
        
        # Create the Tree-sitter parser for the language name
        self.parser = Parser()
        self.parser.set_language(
            Language(INSTALLED_LANGUAGES_FILEPATH, language_name)
        )

    def get_tree(self, source: str):
        """Create a syntax tree based on the grammar and an input string"""
        tree: Tree = self.parser.parse(bytes(source, "utf8"))
        # remove the tree cleaned of empty nodes.
        return self.clean_tree(tree)

    def clean_nodes(self, node: Node):
        """Remove empty children from the node tree"""
        for child in node.children:
            if child.type == '\n': # empty child
                node.children.remove(child)
            else:
                self.clean_nodes(child)
        return node

    def clean_tree(self, tree:Tree):
        """Clean the tree starting at the root node"""
        # prune empty nodes from syntax tree
        clean = Tree
        clean.root_node = self.clean_nodes(tree.root_node )
        return clean

    def print_nodes(self, node: Node, indent = ''):
        """Display the node branch in pretty format"""
        for child in node.children:
            print(f"{indent} node: {child.type}")
            self.print_nodes(child, indent + '  ')

    def print_tree(self, tree: Tree, indent = ''):
        """Display the tree starting at the root node"""
        self.print_nodes(tree.root_node, indent)
