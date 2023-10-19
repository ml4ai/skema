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
    ScalarType
)

from skema.program_analysis.CAST.python.node_helper import (
    NodeHelper,
    get_first_child_by_type,
    get_children_by_types,
    get_first_child_index,
    get_last_child_index,
    get_control_children,
    get_non_control_children
)
from skema.program_analysis.CAST.python.util import (
    generate_dummy_source_refs,
    get_op
)
from skema.program_analysis.CAST.fortran.variable_context import VariableContext

from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH


PYTHON_VERSION = "3.10"

class TS2CAST(object):
    def __init__(self, source_file_path: str, from_file = True):
        # from_file flag is used for testing purposes, when we don't have actual files
        if from_file:
            self.path = Path(source_file_path)
            self.source_file_name = self.path.name
            
            # Python doesn't have a preprocessing step like fortran
            self.source = self.path.read_text()
        else:
            self.path = "None"
            self.source_file_name = "Temp"
            self.source = source_file_path

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

    # TODO: node helper for ignoring comments

    def visit(self, node: Node):
        if node.type == "module":
            return self.visit_module(node)
        elif node.type == "parenthesized_expression":
            # Node for "( op )", extract op
            # The actual op is in the middle of the list of nodes
            return self.visit(node.children[1])
        elif node.type == "expression_statement":
            return self.visit_expression(node)
        elif node.type == "function_definition":
            return self.visit_function_def(node)
        elif node.type == "return_statement":
            return self.visit_return(node)
        elif node.type == "call":
            return self.visit_call(node)
        elif node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type =="unary_operator":
            return self.visit_unary_op(node)
        elif node.type =="binary_operator":
            return self.visit_binary_op(node)
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
                body.append(child_cast)

        self.variable_context.pop_context()
        
        return Module(
            name=None,
            body=body,
            source_refs = [self.node_helper.get_source_ref(node)]
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

    def visit_function_def(self, node: Node) -> FunctionDef:
        ref = self.node_helper.get_source_ref(node)

        name_node = get_first_child_by_type(node, "identifier")
        name = self.visit(name_node)

        # Create new variable context
        self.variable_context.push_context()

        parameters = get_children_by_types(node, "parameters")[0]
        parameters = get_non_control_children(parameters)
        
        # The body of the function is stored in a 'block' type node
        body = get_children_by_types(node, "block")[0].children
        

        func_params = []
        for node in parameters:
            cast = self.visit(node)
            if isinstance(cast, List):
                func_params.extend(cast)
            elif isinstance(cast, AstNode):
                func_params.append(cast)

        func_body = []
        for node in body:
            cast = self.visit(node)
            if isinstance(cast, List):
                func_body.extend(cast)
            elif isinstance(cast, AstNode):
                func_body.append(cast)
            # TODO: Do we need to handle return statements in any special way?

        self.variable_context.pop_context()

        return FunctionDef(
            name=name.val,
            func_args=func_params,
            body=func_body,
            source_refs=[ref]
        )

    def visit_return(self, node: Node) -> ModelReturn:
        ref = self.node_helper.get_source_ref(node)
        ret_val = node.children[1]
        ret_cast = self.visit(ret_val)

        return ModelReturn(value=ret_cast, source_refs=[ref])

    def visit_call(self, node: Node) -> Call:
        ref = self.node_helper.get_source_ref(node)
        func_identifier = get_first_child_by_type(node, "identifier")
        func_name = self.visit(func_identifier) #self.node_helper.get_identifier(func_identifier)

        arg_list = get_first_child_by_type(node, "argument_list")
        args = get_non_control_children(arg_list)

        func_args = []
        for arg in args:
            cast = get_name_node(self.visit(arg))
            if isinstance(cast, List):
                func_args.extend(cast)
            elif isinstance(cast, AstNode):
                func_args.append(cast)

        # Function calls only want the 'Name' part of the 'Var' that the visit returns
        return Call(
            func=func_name.val, 
            arguments=func_args, 
            source_refs=[ref]
        )


    def visit_assignment(self, node: Node) -> Assignment:
        left, _, right = node.children
        ref = self.node_helper.get_source_ref(node)
        
        # For the RHS of an assignment we want the Name CAST node
        # and not the entire Var CAST node if we're doing an
        # assignment like x = y
        right_cast = get_name_node(self.visit(right))
        
        return Assignment(
            left=self.visit(left),
            right=right_cast,
            source_refs=[ref]
        )

    def visit_unary_op(self, node: Node) -> Operator:
        """
            Unary Ops
            OP operand
            where operand is some kind of expression
        """
        ref = self.node_helper.get_source_ref(node)
        op = get_op(self.node_helper.get_operator(node.children[0]))
        operand = node.children[1]
        
        if op == 'ast.Sub':
            op = 'ast.USub'
        
        # For the operand we need the Name CAST node and
        # not the whole Var CAST node
        # in instances like -x
        operand_cast = get_name_node(self.visit(operand))
        
        if isinstance(operand_cast, Var):
            operand_cast = operand_cast.val

        return Operator(
            op=op, 
            operands=[operand_cast], 
            source_refs=[ref]
        )

    def visit_binary_op(self, node: Node) -> Operator:
        """
            Binary Ops
            left OP right
            where left and right can either be operators or literals
        
        """
        ref = self.node_helper.get_source_ref(node)
        op = get_op(self.node_helper.get_operator(node.children[1]))
        left, _, right = node.children

        left_cast = get_name_node(self.visit(left))
        right_cast = get_name_node(self.visit(right))

        return Operator(
            op=op, 
            operands=[left_cast, right_cast], 
            source_refs=[ref]
        )

    def visit_identifier(self, node: Node) -> Var:
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

    def visit_literal(self, node: Node) -> Any:
        literal_type = node.type        
        literal_value = self.node_helper.get_identifier(node)
        literal_source_ref = self.node_helper.get_source_ref(node)

        if literal_type == "integer":
            return LiteralValue(
                value_type=ScalarType.INTEGER,
                value=literal_value,
                source_code_data_type=["Python", PYTHON_VERSION, str(type(1))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "float":
            return LiteralValue(
                value_type=ScalarType.ABSTRACTFLOAT,
                value=literal_value,
                source_code_data_type=["Python", PYTHON_VERSION, str(type(1.0))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "true" or literal_type == "false":
            return LiteralValue(
                value_type=ScalarType.BOOLEAN,
                value="True" if literal_type == "true" else "False",
                source_code_data_type=["Python", PYTHON_VERSION, str(type(True))],
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
            
def get_name_node(node):
    # Given a CAST node, if it's type Var, then we extract the name node out of it
    # If it's anything else, then the node just gets returned normally
    cur_node = node
    if isinstance(node, list):
        cur_node = node[0]

    if isinstance(cur_node, Var):
        return cur_node.val
    else:
        return node