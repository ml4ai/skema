import json
import os.path
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Language, Parser, Node

from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Module,
    SourceRef,
    Assignment,
    LiteralValue,
    Var,
    VarType,
    Name,
    Operator,
    AstNode,
    SourceCodeDataType,
    ModelImport,
    FunctionDef,
    Loop,
    Call,
    ModelReturn,
    ModelIf,
    RecordDef,
    Attribute,
)

from skema.program_analysis.CAST.python.node_helper import (
    NodeHelper,
)
from skema.program_analysis.CAST.fortran.util import generate_dummy_source_refs
from skema.program_analysis.CAST.fortran.variable_context import VariableContext

from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH


PYTHON_VERSION = "3.10"

class PyTS2CAST(object):
    def __init__(self, source_file_path: str):
        self.path = Path(source_file_path)
        self.source_file_name = self.path.name
        
        # Python doesn't have a preprocessing step like fortran
        self.source = self.path.read_text()

        # Run tree-sitter preprocessor output to generate parse tree
        parser = Parser()
        parser.set_language(
            Language(
                INSTALLED_LANGUAGES_FILEPATH,
                "python"
            )
        )

        # Tree walking structures
        self.variable_context = VariableContext()
        self.node_helper = NodeHelper(self.source, self.source_file_name)

        self.tree = parser.parse(bytes(self.source, "utf8"))

        self.out_cast = self.generate_cast()

    def generate_cast(self) -> List[CAST]:
        '''Interface for generating CAST.'''
        module = self.run(self.tree.root_node)
        return CAST([generate_dummy_source_refs(module)], "Python") 
        
    def run(self, root) -> List[Module]:
        # In python there's generally only one module at the top level
        # I believe then we just need to visit the root, which is a module
        # Which can then contain multiple things (Handled at module visitor)
        return self.visit(root)

    def visit(self, node: Node):
        print(f"Visiting node type {node.type}")

        if node.type == "module":
            return self.visit_module(node)
        elif node.type == "expression_statement":
            return self.visit_expression(node)
        elif node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type in ["integer"]:
            return self.visit_literal(node)
        else:
            return self._visit_passthrough(node)

    def visit_module(self, node: Node) -> Module:
        # A module is comprised of one or several statements/expressions
        # At the global level
        self.variable_context.push_context()

        body = []
        for child in node.children:
            child_cast = self.visit(child)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                body.extend(child_cast)

        self.variable_context.pop_context()
        
        return Module(
            name=None,
            body=body,
            source_refs = [self.node_helper.get_source_ref(node)]
        )

    def visit_name(self, node):
        # First, we will check if this name is already defined, and if it is return the name node generated previously
        identifier = self.node_helper.get_identifier(node)
        if self.variable_context.is_variable(identifier):
            return self.variable_context.get_node(identifier)

        return self.variable_context.add_variable(
            identifier, "Unknown", [self.node_helper.get_source_ref(node)]
        )

    def visit_expression(self, node: Node):
        # NOTE: Is there an instance where an 'expression statement' node
        # Has more than one child?

        expr_body = []
        for child in node.children:
            child_cast = self.visit(child)
            if isinstance(child_cast, List):
                expr_body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                expr_body.append(child_cast)

        return expr_body

    def visit_assignment(self, node: Node) -> Assignment:
        left, _, right = node.children
        literal_source_ref = self.node_helper.get_source_ref(node)
        
        return Assignment(
            left=self.visit(left),
            right=self.visit(right),
            source_refs=[literal_source_ref]
        )
        

    def visit_identifier(self, node: Node):
        identifier = self.node_helper.get_identifier(node)

        if self.variable_context.is_variable(identifier):
            var_type = self.variable_context.get_type(identifier)
        else:
            var_type = "unknown"
        
        # TODO: Python default values
        default_value = None
        
        value = self.visit_name(node)
        
        return Var(
            val=value,
            type=var_type,
            default_value=default_value,
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_literal(self, node: Node):
        literal_type = node.type        
        literal_value = self.node_helper.get_identifier(node)
        literal_source_ref = self.node_helper.get_source_ref(node)

        if literal_type == "integer":
            return LiteralValue(
                value_type="Integer",
                value=literal_value,
                source_code_data_type=["Python", PYTHON_VERSION, "integer"],
                source_refs=[literal_source_ref]
            )

    def visit_name(self, node):
        # First, we will check if this name is already defined, and if it is return the name node generated previously
        identifier = self.node_helper.get_identifier(node)
        if self.variable_context.is_variable(identifier):
            return self.variable_context.get_node(identifier)

        return self.variable_context.add_variable(
            identifier, "Unknown", [self.node_helper.get_source_ref(node)]
        )

    def _visit_passthrough(self, node):
        if len(node.children) == 0:
            return None

        for child in node.children:
            child_cast = self.visit(child)
            if child_cast:
                return child_cast