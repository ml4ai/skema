import json
import os.path
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
    ModelBreak,
    ModelContinue,
    ModelIf,
    ModelImport,
    ModelReturn,
    Module,
    Name,
    Operator,
    RecordDef,
    ScalarType,
    SourceCodeDataType,
    SourceRef,
    StructureType,
    ValueConstructor,
    Var,
    VarType,
)
from skema.program_analysis.CAST.matlab.variable_context import (
    VariableContext
)
from skema.program_analysis.CAST.matlab.node_helper import (
    get_children_by_types,
    get_control_children,
    get_first_child_by_type,
    get_keyword_children,
    NodeHelper
)
from skema.program_analysis.CAST.matlab.tokens import KEYWORDS
from skema.program_analysis.tree_sitter_parsers.build_parsers import(
    INSTALLED_LANGUAGES_FILEPATH
)

MATLAB_VERSION='matlab_version_here'
MODULE_NAME='module_name_here'
INTERPRETER='matlab_to_cast'

class MatlabToCast(object):

    node_visits = dict()

    def __init__(self, source_path = "", source = ""):

        # if a source file path is provided, read source from file
        if not source_path == "":
            path = Path(source_path)
            self.filename = path.name
            self.source = path.read_text().strip()
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
        module.name = MODULE_NAME
        self.out_cast = CAST([module], "matlab")

    def log_visit(self, node: Node):
        """ record a visit of a node type """
        key = node.type if node else "None"
        value = self.node_visits[key] + 1 if key in self.node_visits else 1
        self.node_visits[key] = value

    def run(self, root) -> Module:
        """ process an entire Tree-sitter syntax instance """
        return self.visit(root)

    def visit(self, node):
        """Switch execution based on node type"""
        self.log_visit(node)
        if node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "boolean":
            return self.visit_boolean(node)
        elif node.type == "command":
            return self.visit_command(node)
        elif node.type == "function_call":
            return self.visit_function_call(node)
        elif node.type == "function_definition":
            return self.visit_function_def(node)
        elif node.type in [
            "identifier"
        ]:return self.visit_identifier(node)
        elif node.type == "if_statement":
            return self.visit_if_statement(node)
        elif node.type == "iterator":
            return self.visit_iterator(node)
        elif node.type == "for_statement":
            return self.visit_for_statement(node)
        elif node.type in [
            "cell",
            "matrix"
        ]:   return self.visit_matrix(node)
        elif node.type == "source_file":
            return self.visit_module(node)
        elif node.type in [
            "command_name",
            "command_argument",
            "name"
        ]:  return self.visit_name(node)
        elif node.type == "number":
            return self.visit_number(node)
        elif node.type in [
            "binary_operator",
            "boolean_operator",
            "comparison_operator",
            "unary_operator",
            "spread_operator",
            "postfix_operator",
            "not_operator"
        ]: return self.visit_operator(node)
        elif node.type == "string":
           return self.visit_string(node)
        elif node.type == "range":
           return self.visit_range(node)
        elif node.type == "switch_statement":
            return self.visit_switch_statement(node)
        else:
            return self._visit_passthrough(node)

    def visit_assignment(self, node):
        """ Translate Tree-sitter assignment node """
        children = get_keyword_children(node)
        return Assignment(
            left=self.visit(children[0]),
            right=self.visit(children[1]),
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_boolean(self, node):
        """ Translate Tree-sitter boolean node """
        for child in node.children:
            # set the first letter to upper case for python
            value = child.type
            value = value[0].upper() + value[1:].lower()
            # store as string, use Python Boolean capitalization.
            return LiteralValue(
                value_type=ScalarType.BOOLEAN,
                value = value,
                source_code_data_type=["matlab", MATLAB_VERSION, ScalarType.BOOLEAN],
                source_refs=[self.node_helper.get_source_ref(node)],
            )

    def visit_command(self, node):
        """ Translate the Tree-sitter command node """
        children =  get_keyword_children(node)
        argument = [self.visit(children[1])] if len(children) > 1 else []
        return Call(
            func = self.visit(children[0]),
            source_language = "matlab",
            source_language_version = MATLAB_VERSION,
            arguments = argument,
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_function_call(self, node):
        """ Translate Tree-sitter function call node """
        func = self.visit(get_keyword_children(node)[0])
        arguments = [self.visit(child) for child in 
            get_keyword_children(get_first_child_by_type(node, "arguments"))]
        return Call(
            func = func,
            source_language = "matlab",
            source_language_version = MATLAB_VERSION,
            arguments = arguments,
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_function_def(self, node):
        """ return a CAST transation of a MATLAB function definition """
        return FunctionDef(
            name = self.visit(get_first_child_by_type(node, "function_output")),
            body = self.get_block(node),
            func_args = [self.visit(child) for child in 
                get_keyword_children(
                    get_first_child_by_type(node, "function_arguments"))],
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_identifier(self, node):
        """ return an identifier (variable) node """
        identifier = self.node_helper.get_identifier(node)
        return Var(
            val = self.visit_name(node),
            type = self.variable_context.get_type(identifier) if
                self.variable_context.is_variable(identifier) else "Unknown",
            source_refs = [self.node_helper.get_source_ref(node)],
        )

    def visit_if_statement(self, node):
        """ return a node describing if, elseif, else conditional logic"""

        def get_conditional(conditional_node):
            """ return a ModelIf struct for the conditional logic node. """

            # Conditional will be after the "if" or "elseif" child
            for i, child in enumerate(conditional_node.children):
                if child.type in ["if", "elseif"]:
                    expr = conditional_node.children[i+1]

            return ModelIf(
                expr = self.visit(expr),
                body = self.get_block(conditional_node),
                orelse = [],
                source_refs=[self.node_helper.get_source_ref(conditional_node)]
            )

        # start with the if statement
        first = get_conditional(node)
        current = first

        # add 0-n elseif clauses
        for child in get_children_by_types(node, ["elseif_clause"]):
            current.orelse = [get_conditional(child)]
            current = current.orelse[0]

        # add 0-1 else clause 
        else_clause = get_first_child_by_type(node, "else_clause")
        if else_clause:
            current.orelse = self.get_block(else_clause)

        return first

    # CAST has no Iterator node, so we return a partially 
    # completed Loop object 
    # MATLAB iterators are either matrices or ranges.
    def visit_iterator(self, node) -> Loop:

        itr_var = self.visit(get_first_child_by_type(node, "identifier"))
        source_ref = self.node_helper.get_source_ref(node)

        # process matrix iterator
        matrix_node = get_first_child_by_type(node, "matrix")
        if matrix_node is not None:
            row_node = get_first_child_by_type(matrix_node, "row")
            if row_node is not None:
                mat = [self.visit(child) for child in 
                    get_keyword_children(row_node)]
                mat_idx = 0
                mat_len = len(mat)


                return Loop(
                    pre = [
                        Assignment(
                            left = "_mat",
                            right = mat,
                            source_refs = [source_ref]
                        ),
                        Assignment(
                            left = "_mat_len",
                            right = mat_len,
                            source_refs = [source_ref]
                        ),
                        Assignment(
                            left = "_mat_idx",
                            right = mat_idx,
                            source_refs = [source_ref]
                        ),
                        Assignment(
                            left = itr_var,
                            right = mat[mat_idx],
                            source_refs = [source_ref]
                        )
                    ],
                    expr = self.get_operator(
                        op = "<",
                        operands = ["_mat_idx", "_mat_len"],
                        source_refs = [source_ref]
                    ),
                    body = [
                        Assignment(
                            left = "_mat_idx",
                            right = self.get_operator(
                                op = "+",
                                operands = ["_mat_idx", 1],
                                source_refs = [source_ref]
                            ),
                            source_refs = [source_ref]
                        ),
                        Assignment(
                            left = itr_var,
                            right = "_mat[_mat_idx]",
                            source_refs = [source_ref]
                        )
                    ],
                    post = []
                )



        # process range iterator
        range_node = get_first_child_by_type(node, "range")
        if range_node is not None:
            numbers = [self.visit(child) for child in 
                get_children_by_types(range_node, ["number"])]
            start = numbers[0]
            step = 1
            stop = 0
            if len(numbers) == 2:
                stop = numbers[1]

            elif len(numbers) == 3:
                step = numbers[1]
                stop = numbers[2]

            range_name_node = self.variable_context.get_gromet_function_node("range")
            iter_name_node = self.variable_context.get_gromet_function_node("iter")
            next_name_node = self.variable_context.get_gromet_function_node("next")
            generated_iter_name_node = self.variable_context.generate_iterator()
            stop_condition_name_node = self.variable_context.generate_stop_condition()

            return Loop(
                pre = [
                    Assignment(
                        left = itr_var,
                        right = start,
                        source_refs = [source_ref]
                    )
                ],
                expr = self.get_operator(
                    op = "<=",
                    operands = [itr_var, stop],
                    source_refs = [source_ref]
                ),
                body = [
                    Assignment(
                        left = itr_var,
                        right = self.get_operator(
                            op = "+",
                            operands = [itr_var, step],
                            source_refs = [source_ref]
                        ),
                        source_refs = [source_ref]
                    )
                ],
                post = []
            )


    def visit_range(self, node):
        return None

    def visit_for_statement(self, node) -> Loop:
        """ Translate Tree-sitter for loop node into CAST Loop node """

        loop = self.visit(get_first_child_by_type(node, "iterator"))
        loop.source_refs=[self.node_helper.get_source_ref(node)]
        loop.body = self.get_block(node) + loop.body

        return loop


    def visit_matrix(self, node):
        """ Translate the Tree-sitter cell node into a List """

        def get_values(element, ret)-> List:
            for child in get_keyword_children(element):
                if child.type == "row": 
                    ret.append(get_values(child, []))
                else:
                    ret.append(self.visit(child))
            return ret;

        values = get_values(node, [])
        value = []
        if len(values) > 0:
            value = values[0]

        return LiteralValue(
            value_type=StructureType.LIST,
            value = value,
            source_code_data_type=["matlab", MATLAB_VERSION, StructureType.LIST],
            source_refs=[self.node_helper.get_source_ref(node)],
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

    def visit_number(self, node) -> LiteralValue:
        """Visitor for numbers """
        literal_value = self.node_helper.get_identifier(node)
        # Check if this is a real value, or an Integer
        if "e" in literal_value.lower() or "." in literal_value:
            return LiteralValue(
                value_type=ScalarType.ABSTRACTFLOAT,
                value=float(literal_value),
                source_code_data_type=["matlab", MATLAB_VERSION, ScalarType.ABSTRACTFLOAT],
                source_refs=[self.node_helper.get_source_ref(node)]
            )
        return LiteralValue(
            value_type=ScalarType.INTEGER,
            value=int(literal_value),
            source_code_data_type=["matlab", MATLAB_VERSION, ScalarType.INTEGER],
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_operator(self, node):
        """return an operator based on the Tree-sitter node """
        # The operator will be the first control character
        op = self.node_helper.get_identifier(
           get_control_children(node)[0]
        )
        # the operands will be the keyword children
        operands=[self.visit(child) for child in get_keyword_children(node)]
        return self.get_operator(
            op = op,
            operands = operands,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_string(self, node):
        return LiteralValue(
            value_type=ScalarType.CHARACTER,
            value=self.node_helper.get_identifier(node),
            source_code_data_type=["matlab", MATLAB_VERSION, ScalarType.CHARACTER],
            source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_switch_statement(self, node):
        """ return a conditional statement based on a MATLAB switch statement """
        # node types used for case comparison
        case_node_types = [
            "boolean",
            "identifier",
            "matrix",
            "number",
            "string",
            "unary_operator"
        ]

        def get_case_expression(case_node, switch_var):
            """ return an operator representing the case test """
            source_refs=[self.node_helper.get_source_ref(case_node)]
            cell_node = get_first_child_by_type(case_node, "cell")
            # multiple case arguments
            if (cell_node):
                operand = LiteralValue(
                    value_type=StructureType.LIST,
                    value = self.visit(cell_node),
                    source_code_data_type=["matlab", MATLAB_VERSION, StructureType.LIST],
                    source_refs=[self.node_helper.get_source_ref(cell_node)]
                )
                return self.get_operator(
                    op = "in", 
                    operands = [switch_var, operand], 
                    source_refs = source_refs
                )
            # single case argument
            operand = [self.visit(node) for node in 
                get_children_by_types(case_node, case_node_types)][0]
            return self.get_operator(
                op = "==", 
                operands = [switch_var, operand], 
                source_refs = source_refs
            )

        def get_model_if(case_node, switch_var):
            """ return conditional logic representing the case """
            return ModelIf(
                expr = get_case_expression(case_node, switch_var),
                body = self.get_block(case_node),
                orelse = [],
                source_refs=[self.node_helper.get_source_ref(case_node)]
            )
        
        # switch variable is usually an identifier
        switch_var = get_first_child_by_type(node, "identifier")
        if switch_var is not None:
            switch_var = self.visit(switch_var)

        # however it can be a function call
        else:
            switch_var = self.visit(get_first_child_by_type(node, "function_call"))

        # n case clauses as 'if then' nodes
        case_nodes = get_children_by_types(node, ["case_clause"])
        model_ifs = [get_model_if(node, switch_var) for node in case_nodes]
        for i, model_if in enumerate(model_ifs[1:]):
            model_ifs[i].orelse = [model_if]

        # otherwise clause as 'else' node after last 'if then' node
        otherwise_clause = get_first_child_by_type(node, "otherwise_clause")
        if otherwise_clause:
            last = model_ifs[len(model_ifs)-1]
            last.orelse = self.get_block(otherwise_clause)

        return model_ifs[0]
    
    def get_block(self, node) -> List[AstNode]:
        """return all the children of the block as a list of AstNodes"""
        block = get_first_child_by_type(node, "block")
        if block:
            return [self.visit(child) for child in 
                get_keyword_children(block)]

    def get_operator(self, op, operands, source_refs):
        """ return an operator representing the arguments """
        return Operator(
            source_language = "matlab",
            interpreter = INTERPRETER,
            version = MATLAB_VERSION,
            op = op,
            operands = operands,
            source_refs = source_refs
        )

    def get_gromet_function_node(self, func_name: str) -> Name:
        if self.variable_context.is_variable(func_name):
            return self.variable_context.get_node(func_name)

    # skip control nodes and other junk
    def _visit_passthrough(self, node):
        if len(node.children) == 0:
            return []

        for child in node.children:
            child_cast = self.visit(child)
            if child_cast:
                return child_cast
