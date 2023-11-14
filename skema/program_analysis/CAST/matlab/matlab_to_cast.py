import json
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Language, Parser, Node, Tree

from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    AstNode,
    Attribute,
    Call,
    FunctionDef,
    LiteralValue,
    Loop,
    ModelIf,
    ModelImport,
    ModelReturn,
    Module,
    Name,
    Operator,
    RecordDef,
    SourceCodeDataType,
    SourceRef,
    Var,
    VarType,
)

from skema.program_analysis.CAST.matlab.variable_context import VariableContext
from skema.program_analysis.CAST.matlab.node_helper import (
    get_children_by_types,
    get_control_children,
    get_first_child_by_type,
    get_keyword_children,
    NodeHelper
)

from skema.program_analysis.CAST.matlab.tokens import KEYWORDS

from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH

MATLAB_VERSION='matlab_version_here'

class MatlabToCast(object):

    def __init__(self, source_path = "", source = ""):

        # if a source file path is provided, read source from file
        if not source_path == "":
            path = Path(source_path)
            self.source = path.read_text().strip()
            self.filename = path.name
        # otherwise copy the input source and flag the filename unused
        else:
            self.source = source
            self.filename = "None"

        # create MATLAB parser
        parser = Parser()
        parser.set_language(
            Language(INSTALLED_LANGUAGES_FILEPATH, "matlab")
        )
        
        # create a syntax tree using the source file
        self.tree = parser.parse(bytes(self.source, "utf8"))

        # create helper classes
        self.variable_context = VariableContext()
        self.node_helper = NodeHelper(self.source, self.filename)

        # create CAST object 
        module = self.run(self.tree.root_node)
        self.out_cast = CAST([module], "matlab")

    def run(self, root) -> Module:
        return self.visit(root)

    def visit(self, node):
        """Switch execution based on node type"""
        if node == None:
            return None
        elif node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "command":
            return self.visit_command(node)
        elif node.type == "function_call":
            return self.visit_function_call(node)
        elif node.type == "function_definition":
            return self.visit_function_def(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type == "if_statement":
            return self.visit_if_statement(node)
        elif node.type in [
            "boolean",
            "matrix",
            "number",
            "string"
        ]: return self.visit_literal(node)
        elif node.type in [
            "for_statement",
            "iterator",
            "while_statement",
            "spread_operator"
        ]: return self.visit_loop(node)
        elif node.type == "source_file":    # used?
            return self.visit_module(node)
        elif node.type == "name":
            return self.visit_name(node)
        elif node.type in [
            "binary_operator",
            "comparison_operator",
            "boolean_operator"
        ]: return self.visit_operator_binary(node)
        elif node.type == "not_operator":
            return self.visit_operator_not(node)
        elif node.type == "postfix_operator":
            return self.visit_operator_postfix(node)
        elif node.type == "unary_operator":
            return self.visit_operator_unary(node)
        elif node.type == "switch_statement":
            return self.visit_switch_statement(node)
        else:
            return self._visit_passthrough(node)

    def visit_assignment(self, node):
        """ Translate Tree-sitter assignment node """
        left, _, right = node.children
        return Assignment(
            left=self.visit(left),
            right=self.visit(right),
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_command(self, node):
        """ Translate the Tree-sitter command node """
        command_name = get_first_child_by_type(node, "command_name")
        command_argument = get_first_child_by_type(node, "command_argument")
        return Call(
            func = self.visit(command_name),
            source_language = "matlab",
            source_language_version = MATLAB_VERSION,
            arguments = [self.visit(command_argument)]
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_function_call(self, node):
        """ Translate Tree-sitter function call node """
        args_parent = get_first_child_by_type(node, "arguments")
        args_children = [c for c in get_keyword_children(args_parent)]
        return Call(
            func = self.visit(get_first_child_by_type(node, "identifier")),
            source_language = "matlab",
            source_language_version = MATLAB_VERSION,
            arguments = [self.visit(c) for c in args_children],
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_function_def(self, node):

        # ...

        return FunctionDef(
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_identifier(self, node):
        """ return an identifier (variable) node """
        identifier = self.node_helper.get_identifier(node)
        if self.variable_context.is_variable(identifier):
            var_type = self.variable_context.get_type(identifier)
        else:
            var_type = "Unknown"

        default_value = None

        value = self.visit_name(node)

        return Var(
            val=value,
            type=var_type,
            default_value=default_value,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_if_statement(self, node):
        """ return a node describing if, elseif, else conditional logic"""
        def conditional(conditional_node):
            """ return a ModelIf struct for the conditional logic node. """
            ret = ModelIf()
            # comparison_operator
            ret.expr = self.visit(get_first_child_by_type(
                conditional_node,
                "comparison_operator"
            ))
            # instruction_block
            block = get_first_child_by_type(conditional_node, "block")
            if block:
                children = get_keyword_children(block)
                ret.body = [self.visit(child) for child in children]

            return ret

        # the if statement is returned as a ModelIf AstNode
        model_ifs = [conditional(node)]

        # add 0-n elseif clauses 
        elseif_clauses = get_children_by_types(node, ["elseif_clause"])
        model_ifs += [conditional(child) for child in elseif_clauses]
        for i, model_if in enumerate(model_ifs[1:]):
            model_ifs[i].orelse = [model_if]

        # add 0-1 else clause 
        else_clause = get_first_child_by_type(node, "else_clause")
        if else_clause:
            block = get_first_child_by_type(else_clause, "block")
            if block:
                last = model_ifs[len(model_ifs)-1]
                children = get_keyword_children(block)
                last.orelse = [self.visit(child) for child in children]

        return model_ifs[0]

    def visit_literal(self, node) -> LiteralValue:
        """Visitor for literals. Returns a LiteralValue"""
        literal_type = node.type
        literal_value = self.node_helper.get_identifier(node)
        literal_source_ref = self.node_helper.get_source_ref(node)

        if literal_type == "number":
            # Check if this is a real value, or an Integer
            if "e" in literal_value.lower() or "." in literal_value:
                return LiteralValue(
                    value_type="AbstractFloat",  # TODO verify this value
                    value=float(literal_value),
                    source_code_data_type=["matlab", MATLAB_VERSION, "real"],
                    source_refs=[literal_source_ref],
                )
            else:
                return LiteralValue(
                    value_type="Integer",
                    value=int(literal_value),
                    source_code_data_type=["matlab", MATLAB_VERSION, "integer"],
                    source_refs=[literal_source_ref],
                )

        elif literal_type == "string":
            return LiteralValue(
                value_type="Character",
                value=literal_value,
                source_code_data_type=["matlab", MATLAB_VERSION, "character"],
                source_refs=[literal_source_ref],
            )

        elif literal_type == "boolean":
            # store as string, use Python boolean capitalization.
            value = literal_value[0].upper() + literal_value[1:].lower() 
            return LiteralValue(
                value_type="Boolean",
                value = value,
                source_code_data_type=["matlab", MATLAB_VERSION, "logical"],
                source_refs=[literal_source_ref],
            )

        elif literal_type == "matrix":
            def get_values(values, ret)-> List:
                for child in get_keyword_children(values):
                    if child.type == "row": 
                        ret.append(get_values(child, []))
                    else:
                        ret.append(self.visit(child))
                return ret;

            matrix_values = get_values(node, [])
            value = []
            if len(matrix_values) > 0:
                value = matrix_values[0]

            return LiteralValue(
                value_type="List",
                value = value,
                source_code_data_type=["matlab", MATLAB_VERSION, "matrix"],
                source_refs=[literal_source_ref],
            )

    # General loop translator for all MATLAB loop types
    def visit_loop(self, node) -> Loop:
        """ Translate Tree-sitter for_loop node into CAST Loop node """
        return Loop (
            source_refs = [self.node_helper.get_source_ref(node)]
        )

    def visit_module(self, node: Node) -> Module:
        """Visitor for program and module statement. Returns a Module object"""
        self.variable_context.push_context()
        
        program_body = []
        for child in node.children: 
            child_cast = self.visit(child)
            if isinstance(child_cast, List):
                program_body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                program_body.append(child_cast)
    
        self.variable_context.pop_context()
        
        return Module(
            name=None, #TODO: Fill out name field
            body=program_body,
            source_refs = [self.node_helper.get_source_ref(node)]
        )

    def visit_name(self, node):
        """ return or create the node for this variable name """
        identifier = self.node_helper.get_identifier(node)
        # if the identifier exists, return its node
        if self.variable_context.is_variable(identifier):
            return self.variable_context.get_node(identifier)
        # create a new node
        return self.variable_context.add_variable(
            identifier, "Unknown", [self.node_helper.get_source_ref(node)]
        )

    def visit_operator_binary(self, node):
        op = self.node_helper.get_identifier(
            get_control_children(node)[0]
        )  # The operator will be the first control character

        operands = []
        for operand in get_keyword_children(node):
            operands.append(self.visit(operand))

        return Operator(
            source_language="matlab",
            interpreter=None,
            version=None,
            op=op,
            operands=operands,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    # TODO implement as nested for loops
    def visit_operator_not(self, node):
        # The MATLAB not operator is a logical matrix inversion
        return None

    # TODO
    def visit_operator_postfix(self, node):
        return None

    def visit_operator_unary(self, node):
        # A unary operator is an Operator instance with a single operand
        return Operator(
            source_language="matlab",
            interpreter=None,
            version=None,
            op = node.children[0].type,
            operands=[self.visit(node.children[1])],
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_switch_statement(self, node):
        """ return a conditional statement based on a MATLAB switch statement """
        # node types used for case comparison
        case_node_types = [
            "boolean",
            "identifier",
            "number",
            "string",
            "unary_operator"
        ]
        
        def get_node_value(ast_node):
            """ return the CAST node value or var name """
            if isinstance(ast_node, Var):
                return ast_node.val.name
            return ast_node.value

        def get_operator(op, operands, source_refs):
            """ return an Operator representing the case test """
            return Operator(
                source_language = "matlab",
                interpreter = None,
                version = MATLAB_VERSION,
                op = op,
                operands = operands,
                source_refs = source_refs
            )

        def get_case_expression(case_node, identifier):
            """ return an Operator representing the case test """
            source_refs=[self.node_helper.get_source_ref(case_node)]
            cell_node = get_first_child_by_type(case_node, "cell")
            # multiple case arguments
            if (cell_node):
                nodes = get_children_by_types(cell_node, case_node_types)
                operand = LiteralValue(
                    value_type="List",
                    value = [self.visit(node) for node in nodes],
                    source_code_data_type=["matlab", MATLAB_VERSION, "unknown"],
                    source_refs=[self.node_helper.get_source_ref(cell_node)]
                )
                return get_operator("in", [identifier, operand], source_refs)
            # single case argument
            nodes = get_children_by_types(case_node, case_node_types)
            operand = [self.visit(node) for node in nodes][0]
            return get_operator("==", [identifier, operand], source_refs)

        def get_case_body(case_node):
            """ return the instruction block for the case """
            block = get_first_child_by_type(case_node, "block")
            if block:
                return [self.visit(c) for c in get_keyword_children(block)]
            return None
            
        def get_model_if(case_node, identifier):
            """ return conditional logic representing the case """
            return ModelIf(
                expr = get_case_expression(case_node, identifier),
                body = get_case_body(case_node),
                source_refs=[self.node_helper.get_source_ref(case_node)]
            )
        
        # switch statement identifier
        identifier = self.visit(get_first_child_by_type(node, "identifier"))
        
        # n case clauses as 'if then' nodes
        case_nodes = get_children_by_types(node, ["case_clause"])
        model_ifs = [get_model_if(node, identifier) for node in case_nodes]
        for i, model_if in enumerate(model_ifs[1:]):
            model_ifs[i].orelse = [model_if]

        # otherwise clause as 'else' node after last 'if then' node
        otherwise_clause = get_first_child_by_type(node, "otherwise_clause")
        if otherwise_clause:
            block = get_first_child_by_type(otherwise_clause, "block")
            if block:
                last = model_ifs[len(model_ifs)-1]
                last.orelse = [self.visit(c) for c in get_keyword_children(block)]
        return model_ifs[0]

    # this is not in the Waterloo model but will no doubt be encountered.
    def visit_import_statemement(self, node):
        # (use)
        #   (use)
        #   (module_name)

        ## Pull relevent child nodes
        module_name_node = get_first_child_by_type(node, "module_name")
        module_name = self.node_helper.get_identifier(module_name_node)
        included_items_node = get_first_child_by_type(node, "included_items")

        import_all = included_items_node is None
        import_alias = None  # TODO: Look into local-name and use-name fields

        # We need to check if this import is a full import of a module, i.e. use module
        # Or a partial import i.e. use module,only: sub1, sub2
        if import_all:
            return ModelImport(
                name=module_name,
                alias=import_alias,
                all=import_all,
                symbol=None,
                source_refs=None,
            )
        else:
            imports = []
            for symbol in get_keyword_children(included_items_node):
                symbol_identifier = self.node_helper.get_identifier(symbol)
                symbol_source_refs = [self.node_helper.get_source_ref(symbol)]
                imports.append(
                    ModelImport(
                        name=module_name,
                        alias=import_alias,
                        all=import_all,
                        symbol=symbol_identifier,
                        source_refs=symbol_source_refs,
                    )
                )

            return imports

    # skip control nodes and other junk
    def _visit_passthrough(self, node):
        if len(node.children) == 0:
            return None

        for child in node.children:
            child_cast = self.visit(child)
            if child_cast:
                return child_cast
