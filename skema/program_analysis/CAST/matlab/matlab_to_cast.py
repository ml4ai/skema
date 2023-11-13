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
    get_all,
    get_children_by_types,
    get_control_children,
    get_first_child_by_type,
    get_first_child_index,
    get_keyword_children,
    get_last_child_index,
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
        modules = self.run(self.tree.root_node)
        self.out_cast =  [CAST([module], "matlab") for module in modules]

    def run(self, root) -> List[Module]:
        return [self.visit(root)]

    def visit(self, node):
        """Switch execution based on node type"""
        if node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "command":
            return self.visit_command(node)
        elif node.type == "for_statement":
            return self.visit_for_statement(node)
        elif node.type == "function_call":
            return self.visit_function_call(node)
        elif node.type == "function_definition":
            return self.visit_function_def(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type == "if_statement":
            return self.visit_if_statement(node)
        elif node.type == "iterator":    # used?
            return self.visit_iterator(node)
        elif node.type in [
            "boolean",
            "matrix",
            "number",
            "string"
        ]: return self.visit_literal(node)
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
        elif node.type == "spread_operator":
            return self.visit_operator_spread(node)
        elif node.type == "unary_operator":
            return self.visit_operator_unary(node)
        elif node.type == "row":
            return self.visit_row(node)
        elif node.type == "switch_statement":
            return self.visit_switch_statement(node)
        elif node.type == "while_statement":
            return self.visit_while_statement(node)
        else:
            return self._visit_passthrough(node)

    def visit_assignment(self, node):
        left, _, right = node.children

        return Assignment(
            left=self.visit(left),
            right=self.visit(right),
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_command(self, node):
        # Pull relevent nodes
        print ("visit_command")
        print(node)
        return None

    # Note that this is a wrapper for an iterator
    def visit_for_statement(self, node) -> Loop:
        """ Translate Tree-sitter for_loop node into CAST Loop node """
        """
        SOURCE:
        for n = 1:10:2
            x = step_by_2(n)
        end

        SYNTAX TREE:
        for_statement
            iterator
                identifier n
                =
                range
                    number  % start
                    :
                    number  % stop
                    :
                    number  % step
            block
                assignment
                    identifier x
                    =
                    function_call
                        identifier step_by_2
                        (
                        arguments
                            identifier n
                        )
            end
        ;
        """

        """
        class Loop(AstNode):
            'pre': 'list[AstNode]',
            'expr': 'AstNode',
            'body': 'list[AstNode]',
            'post': 'list[AstNode]'
        """

        iterator_node = get_first_child_by_type(node, "iterator")
        range_var = get_first_child_by_type(iterator_node, "identifier")
        range_node = get_first_child_by_type(iterator_node, "range")
        range_children = get_keyword_children(range_node) + [None]
        range_start, range_stop, range_step = range_children

        return Loop(
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_function_call(self, node):
        """
        SOURCE:
        subplot(3,5,7);

        SYNTAX TREE:
        function_call
            identifier
            (
            arguments
                number
                ,
                number
                ,
                number
            )
        ;
        """

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
        # TODO: Refactor function def code to use new helper functions
        # Node structure
        # (subroutine)
        #   (subroutine_statement)
        #     (subroutine)
        #     (name)
        #     (parameters) - Optional
        #   (body_node) ...
        # (function)
        #   (function_statement)
        #     (function)
        #     (intrinsic_type) - Optional
        #     (name)
        #     (parameters) - Optional
        #     (function_result) - Optional
        #       (identifier)
        #  (body_node) ...

        # Create a new variable context
        self.variable_context.push_context()

        # Top level statement node
        statement_node = get_children_by_types(node, ["function_definition"])[0]
        name_node = get_first_child_by_type(statement_node, "name")
        name = self.visit(
            name_node
        )  # Visit the name node to add it to the variable context

        # If this is a function, check for return type and return value
        intrinsic_type = None
        return_value = None
        if node.type == "function_definition":
            signature_qualifiers = get_children_by_types(
                statement_node, ["intrinsic_type", "function_result"]
            )
            for qualifier in signature_qualifiers:
                if qualifier.type == "intrinsic_type":
                    intrinsic_type = self.node_helper.get_identifier(qualifier)
                    self.variable_context.add_variable(
                        self.node_helper.get_identifier(name_node), intrinsic_type, None
                    )
                elif qualifier.type == "function_result":
                    return_value = self.visit(
                        get_first_child_by_type(qualifier, "identifier")
                    )  # TODO: UPDATE NODES
                    self.variable_context.add_return_value(return_value.val.name)

        # #TODO: What happens if function doesn't return anything?
        # If this is a function, and there is no explicit results variable, then we will assume the return value is the name of the function
        if not return_value:
            self.variable_context.add_return_value(
                self.node_helper.get_identifier(name_node)
            )

        # If funciton has both, then we also need to update the type of the return value in the variable context
        # It does not explicity have to be declared
        if return_value and intrinsic_type:
            self.variable_context.update_type(return_value.val.name, intrinsic_type)

        # Generating the function arguments by walking the parameters node
        func_args = []
        if parameters_node := get_first_child_by_type(statement_node, "parameters"):
            for parameter in get_keyword_children(parameters_node):
                # For both subroutine and functions, all arguments are assumes intent(inout) by default unless otherwise specified with intent(in)
                # The variable declaration visitor will check for this and remove any arguments that are input only from the return values
                self.variable_context.add_return_value(
                    self.node_helper.get_identifier(parameter)
                )
                func_args.append(self.visit(parameter))

        # The first child of function will be the function statement, the rest will be body nodes
        body = []
        for body_node in node.children[1:]:
            child_cast = self.visit(body_node)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                body.append(child_cast)

        # After creating the body, we can go back and update the var nodes we created for the arguments
        # We do this by looking for intent,in nodes
        for i, arg in enumerate(func_args):
            func_args[i].type = self.variable_context.get_type(arg.val.name)

        # Pop variable context off of stack before leaving this scope
        self.variable_context.pop_context()

        return FunctionDef(
            name=name,
            func_args=func_args,
            body=body,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_identifier(self, node):
        # By default, this is unknown, but can be updated by other visitors
        identifier = self.node_helper.get_identifier(node)
        if self.variable_context.is_variable(identifier):
            var_type = self.variable_context.get_type(identifier)
        else:
            var_type = "Unknown"

        # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
        # Fortran does have optional arguments introduced in F90, but these do not specify a default
        default_value = None

        # This is another case where we need to override the visitor to explicitly visit another node
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

    # Handle the MATLAB iterator.   
    # Note that this is a wrapper for an iterator
    def visit_iterator(self, node) -> Loop:
        """
        SOURCE:
        for n = 1:10:2
            x = step_by_2(n)
        end

        SYNTAX TREE:
        for_statement
            iterator
                identifier n
                =
                range
                    number  % start
                    :
                    number  % stop
                    :
                    number  % step
            block
                assignment
                    identifier x
                    =
                    function_call
                        identifier step_by_2
                        (
                        arguments
                            identifier n
                        )
            end
        ;
        """

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
                    value=literal_value,
                    source_code_data_type=["matlab", MATLAB_VERSION, "real"],
                    source_refs=[literal_source_ref],
                )
            else:
                return LiteralValue(
                    value_type="Integer",
                    value=literal_value,
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
            return LiteralValue(
                value_type="Boolean",
                value=literal_value,
                source_code_data_type=["matlab", MATLAB_VERSION, "logical"],
                source_refs=[literal_source_ref],
            )

        elif literal_type == "matrix":
            elements = []
            for child in get_keyword_children(node):
                elements.append(self.visit(child))
            return LiteralValue(
                value_type="List",
                value = elements,
                source_code_data_type=["matlab", MATLAB_VERSION, "matrix"],
                source_refs=[literal_source_ref],
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
        # Node structure
        # (name)

        # First, we will check if this name is already defined, and if it is return the name node generated previously
        identifier = self.node_helper.get_identifier(node)
        if self.variable_context.is_variable(identifier):
            return self.variable_context.get_node(identifier)

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

    # TODO implement as for loop
    def visit_operator_spread(self, node):
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

    def visit_operator_row(self, node):
        return None

    def visit_switch_statement(self, node):
        """ return a conditional statement based on the switch statement """
        """
        SOURCE:
        switch x
            case 'one'
                n = 1;
            otherwise
                n = 0;
        end

        SYNTAX TREE:
        switch_statement
            identifier
            case_clause
                case
                string
                    string_content
                block
                    assignment
                        identifier
                        =
                        number
            otherwise_clause
                otherwise
                block
                    assignment
                        identifier
                        =
                        number
            end
        """
    
        # node types used for case comparison
        case_node_types = ["number", "string", "boolean","identifier"]
        
        def get_node_value(ast_node):
            """ return the CAST node value or var name """
            if isinstance(ast_node, Var):
                return ast_node.val.name
            return ast_node.value

        def get_operator(op, operands):
            """ return an Operator representing the case test """
            return Operator(
                source_language = "matlab",
                interpreter = None,
                version = MATLAB_VERSION,
                op = op,
                operands = operands
            )

        def get_case_expression(case_node, identifier):
            """ return an Operator representing the case test """
            cell_node = get_first_child_by_type(case_node, "cell")
            # multiple case arguments
            if (cell_node):
                nodes = get_all(cell_node, case_node_types)
                ast_nodes = [self.visit(node) for node in nodes]
                operand = LiteralValue(
                    value_type="List",
                    value = [get_node_value(node) for node in ast_nodes],
                    source_code_data_type=["matlab", MATLAB_VERSION, "unknown"],
                    source_refs=[self.node_helper.get_source_ref(cell_node)]
                )
                return get_operator("in", [identifier, operand])
            # single case argument
            nodes = get_children_by_types(case_node, case_node_types)
            operand = [self.visit(node) for node in nodes][0]
            return get_operator("==", [identifier, operand])

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

    # into a CAST-supported loop type.
    def visit_while_statement(self, node) -> Loop:
        """ Translate MATLAB while_loop syntax node into CAST Loop node """
        """
        SOURCE:
        n = 10;
        f = n;
        while n > 1
            n = n-1;
            f = f*n;
        end

        SYNTAX TREE:
        assignment
            identifier
            =
            number
        ;
        assignment
            identifier
            =
            identifier
        ;
        while_statement
            while
            comparison_operator
                identifier
                >
                number
            block
                assignment
                    identifier
                    =
                    binary_operator
                        identifier
                        -
                        number
                ;
                assignment
                    identifier
                    =
                    binary_operator
                        identifier
                        *
                        identifier
                ;
            end
        ;
        """

        return Loop(
            source_refs=[self.node_helper.get_source_ref(node)],
        )

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
