from tree_sitter import Node, Tree


class TreeUtils():

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
