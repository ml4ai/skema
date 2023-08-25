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

from skema.program_analysis.CAST.fortran.variable_context import VariableContext
from skema.program_analysis.CAST.fortran.node_helper import (
    NodeHelper,
    get_children_by_types,
    get_first_child_by_type,
    get_control_children,
    get_non_control_children,
    get_first_child_index,
    get_last_child_index,
)
from skema.program_analysis.CAST.fortran.util import generate_dummy_source_refs

from skema.program_analysis.CAST.fortran.preprocessor.preprocess import preprocess
from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH

class TS2CAST(object):
    def __init__(self, source_file_path: str):
        # Prepare source with preprocessor
        self.path = Path(source_file_path)
        self.source_file_name = self.path.name
        self.source = preprocess(self.path)
        
        # Run tree-sitter on preprocessor output to generate parse tree
        parser = Parser()
        parser.set_language(
            Language(
                INSTALLED_LANGUAGES_FILEPATH,
                "fortran"
            )
        )
        self.tree = parser.parse(bytes(self.source, "utf8"))

        # Walking data
        self.variable_context = VariableContext()
        self.node_helper = NodeHelper(self.source, self.source_file_name)

        # Start visiting
        self.out_cast = self.generate_cast()
    def generate_cast(self) -> List[CAST]:
        '''Interface for generating CAST.'''
        modules = self.run(self.tree.root_node)
        return [CAST([generate_dummy_source_refs(module)], "Fortran") for module in modules]
        
    def run(self, root) -> List[Module]:
        '''Top level visitor function. Will return between 1-3 Module objects.'''
        # A program can have between 1-3 modules
        # 1. A module body
        # 2. A program body
        # 3. Everything else (defined functions)
        modules = []

        contexts = get_children_by_types(root, ["module", "program"])
        for context in contexts:
            modules.append(self.visit(context))

        # Currently, we are supporting functions and subroutines defined outside of programs and modules
        # Other than comments, it is unclear if anything else is allowed.
        # TODO: Research the above
        outer_body_nodes = get_children_by_types(root, ["function", "subroutine"])
        if len(outer_body_nodes) > 0:
            body = []
            for body_node in outer_body_nodes:
                child_cast = self.visit(body_node)
                if isinstance(child_cast, List):
                    body.extend(child_cast)
                elif isinstance(child_cast, AstNode):
                    body.append(child_cast)
            modules.append(Module(
                name=None,
                body=body,
                source_refs=[self.node_helper.get_source_ref(root)]
            ))
    
        return modules

    def visit(self, node: Node):
        if node.type in ["program", "module"] :
            return self.visit_module(node)
        elif node.type == "internal_procedures":
            return self.visit_internal_procedures(node)
        elif node.type in ["subroutine", "function"]:
            return self.visit_function_def(node)
        elif node.type in ["subroutine_call", "call_expression"]:
            return self.visit_function_call(node)
        elif node.type == "use_statement":
            return self.visit_use_statement(node)
        elif node.type == "variable_declaration":
            return self.visit_variable_declaration(node)
        elif node.type == "assignment_statement":
            return self.visit_assignment_statement(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type == "name":
            return self.visit_name(node)
        elif node.type in ["math_expression", "relational_expression"]:
            return self.visit_math_expression(node)
        elif node.type in [
            "number_literal",
            "array_literal",
            "string_literal",
            "boolean_literal",
        ]:
            return self.visit_literal(node)
        elif node.type == "keyword_statement":
            return self.visit_keyword_statement(node)
        elif node.type == "extent_specifier":
            return self.visit_extent_specifier(node)
        elif node.type == "do_loop_statement":
            return self.visit_do_loop_statement(node)
        elif node.type == "if_statement":
            return self.visit_if_statement(node)
        elif node.type == "logical_expression":
            return self.visit_logical_expression(node)
        elif node.type == "derived_type_definition":
            return self.visit_derived_type(node)
        elif node.type == "derived_type_member_expression":
            return self.visit_derived_type_member_expression(node)
        else:
            return self._visit_passthrough(node)

    def visit_module(self, node: Node) -> Module:
        '''Visitor for program and module statement. Returns a Module object'''
        self.variable_context.push_context()
        
        program_body = []
        for child in node.children[1:-1]:  # Ignore the start and end program statement
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

    def visit_internal_procedures(self, node: Node) -> List[FunctionDef]:
        '''Visitor for internal procedures. Returns list of FunctionDef'''
        internal_procedures = get_children_by_types(node, ["function", "subroutine"])
        return [self.visit(procedure) for procedure in internal_procedures]

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
        statement_node = get_children_by_types(node, ["subroutine_statement", "function_statement"])[0]
        name_node = get_first_child_by_type(statement_node, "name")
        name = self.visit(
            name_node
        )  # Visit the name node to add it to the variable context

        # If this is a function, check for return type and return value
        intrinsic_type = None
        return_value = None
        if node.type == "function":
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
            for parameter in get_non_control_children(parameters_node):
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

        # TODO:
        # This logic can be made cleaner
        # Fortran doesn't require a return statement, so we need to check if there is a top-level return statement
        # If there is not, then we will create a dummy one
        return_found = False
        for child in body:
            if isinstance(child, ModelReturn):
                return_found = True
        if not return_found:
            body.append(self.visit_keyword_statement(node))

        # Pop variable context off of stack before leaving this scope
        self.variable_context.pop_context()

        return FunctionDef(
            name=name,
            func_args=func_args,
            body=body,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_function_call(self, node):
        # Pull relevent nodes
        if node.type == "subroutine_call":
            function_node = node.children[1]
            arguments_node = node.children[2]
        elif node.type == "call_expression":
            function_node = node.children[0]
            arguments_node = node.children[1]

        function_identifier = self.node_helper.get_identifier(function_node)

        # Tree-Sitter incorrectly parses mutlidimensional array accesses as function calls
        # We will need to check if this is truly a function call or a subscript
        if self.variable_context.is_variable(function_identifier):
            if self.variable_context.get_type(function_identifier) == "List":
                return self._visit_get(
                    node
                )  # This overrides the visitor and forces us to visit another

        # TODO: What should get a name node? Instrincit functions? Imported functions?
        # Judging from the Gromet generation pipeline, it appears that all functions need Name nodes.
        if self.variable_context.is_variable(function_identifier):
            func = self.variable_context.get_node(function_identifier)
        else:
            func = Name(function_identifier, -1)  # TODO: REFACTOR

        # Add arguments to arguments list
        arguments = []
        for argument in arguments_node.children:
            child_cast = self.visit(argument)
            if child_cast:
                arguments.append(child_cast)

        return Call(
            func=func,
            source_language="Fortran",
            source_language_version="2008",
            arguments=arguments,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_keyword_statement(self, node):
        # Currently, the only keyword_identifier produced by tree-sitter is Return
        # However, there may be other instances

        # In Fortran the return statement doesn't return a value (there is the obsolete "alternative return")
        # We keep track of values that need to be returned in the variable context
        return_values = self.variable_context.context_return_values[
            -1
        ]  # TODO: Make function for this

        if len(return_values) == 1:
            # TODO: Fix this case
            value = self.variable_context.get_node(list(return_values)[0])
        elif len(return_values) > 1:
            value = LiteralValue(
                value_type="Tuple",
                value=[
                    Var(
                        val=self.variable_context.get_node(ret),
                        type=self.variable_context.get_type(ret),
                        default_value=None,
                        source_refs=None,
                    )
                    for ret in return_values
                ],
                source_code_data_type=None,  # TODO: REFACTOR
                source_refs=None,
            )
        else:
            value = LiteralValue(val=None, type=None, source_refs=None)

        return ModelReturn(
            value=value, source_refs=[self.node_helper.get_source_ref(node)]
        )

    def visit_use_statement(self, node):
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
            for symbol in get_non_control_children(included_items_node):
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

    def visit_do_loop_statement(self, node) -> Loop:
        """Visitor for Loops. Do to complexity, this visitor logic only handles the range-based do loop.
        The do while loop will be passed off to a seperate visitor. Returns a Loop object.
        """
        """
        Node structure
        Do loop
        (do_loop_statement)
            (loop_control_expression)
                (...) ...
            (body) ...
        
        Do while
        (do_loop_statement)
            (while_statement)
                (parenthesized_expression)
                    (...) ...
            (body) ...
        """

        # First check for
        # TODO: Add do until Loop support
        while_statement_node = get_first_child_by_type(node, "while_statement")
        if while_statement_node:
            return self._visit_while(node)

        # The first body node will be the node after the loop_control_expression
        # NOTE: This code is for the creation of the main body. The do loop will still add some additional nodes at the end of this body.
        body = []
        body_start_index = 1 + get_first_child_index(node, "loop_control_expression")
        for body_node in node.children[body_start_index:]:
            child_cast = self.visit(body_node)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                body.append(child_cast)

        # For the init and expression fields, we first need to determine if we are in a regular "do" or a "do while" loop
        # PRE:
        # _next(_iter(range(start, stop, step)))
        loop_control_node = get_first_child_by_type(node, "loop_control_expression")
        loop_control_children = get_non_control_children(loop_control_node)
        if len(loop_control_children) == 3:
            itterator, start, stop = [
                self.visit(child) for child in loop_control_children
            ]
            step = LiteralValue("Integer", "1")
        elif len(loop_control_children) == 4:
            itterator, start, stop, step = [
                self.visit(child) for child in loop_control_children
            ]
        else:
            itterator = None
            start = None
            stop = None
            step = None

        range_name_node = self.get_gromet_function_node("range")
        iter_name_node = self.get_gromet_function_node("iter")
        next_name_node = self.get_gromet_function_node("next")
        generated_iter_name_node = self.variable_context.generate_iterator()
        stop_condition_name_node = self.variable_context.generate_stop_condition()

        # generated_iter_0 = iter(range(start, stop, step))
        pre = []
        pre.append(
            Assignment(
                left=Var(generated_iter_name_node, "Iterator"),
                right=Call(
                    iter_name_node,
                    arguments=[Call(range_name_node, arguments=[start, stop, step])],
                ),
            )
        )

        # (i, generated_iter_0, sc_0) = next(generated_iter_0)
        pre.append(
            Assignment(
                left=LiteralValue(
                    "Tuple",
                    [
                        itterator,
                        Var(generated_iter_name_node, "Iterator"),
                        Var(stop_condition_name_node, "Boolean"),
                    ],
                ),
                right=Call(
                    next_name_node,
                    arguments=[Var(generated_iter_name_node, "Iterator")],
                ),
            )
        )

        # EXPR
        expr = []
        expr = Operator(
            op="!=",  # TODO: Should this be == or !=
            operands=[
                Var(stop_condition_name_node, "Boolean"),
                LiteralValue("Boolean", True),
            ],
        )

        # BODY
        # At this point, the body nodes have already been visited
        # We just need to append the iterator next call
        body.append(
            Assignment(
                left=LiteralValue(
                    "Tuple",
                    [
                        itterator,
                        Var(generated_iter_name_node, "Iterator"),
                        Var(stop_condition_name_node, "Boolean"),
                    ],
                ),
                right=Call(
                    next_name_node,
                    arguments=[Var(generated_iter_name_node, "Iterator")],
                ),
            )
        )

        # POST
        post = []
        post.append(
            Assignment(
                left=itterator,
                right=Operator(op="+", operands=[itterator, step]),
            )
        )

        return Loop(
            pre=pre,
            expr=expr,
            body=body,
            post=post,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_if_statement(self, node):
        # (if_statement)
        #  (if)
        #  (parenthesised_expression)
        #  (then)
        #  (body_nodes) ...
        #  (elseif_clauses) ..
        #  (else_clause)
        #  (end_if_statement)

        if_condition = self.visit(get_first_child_by_type(node, "parenthesized_expression"))

        child_types = [child.type for child in node.children]

        try:
            elseif_index = child_types.index("elseif_clause")
        except ValueError:
            elseif_index = -1

        try:
            else_index = child_types.index("else_clause")
        except ValueError:
            else_index = -1

        if elseif_index != -1:
            body_stop_index = elseif_index
        else:
            body_stop_index = else_index

        prev = None
        orelse = None
        # If there are else_if statements, they need
        if elseif_index != -1:
            orelse = ModelIf()
            prev = orelse
            for condition in node.children[elseif_index:else_index]:
                if condition.type == "comment":
                    continue
                elseif_expr = self.visit(condition.children[2])
                elseif_body = [self.visit(child) for child in condition.children[4:]]

                prev.orelse = ModelIf(elseif_expr, elseif_body, [])
                prev = prev.orelse

        if else_index != -1:
            else_body = [
                self.visit(child) for child in node.children[else_index].children[1:]
            ]
            if prev:
                prev.orelse = else_body
            else:
                orelse = else_body

        if isinstance(orelse, ModelIf):
            orelse = orelse.orelse

        return ModelIf(
            expr=self.visit(node.children[1]),
            body=[self.visit(child) for child in node.children[3:body_stop_index]],
            orelse=[orelse] if orelse else [],
        )

    def visit_logical_expression(self, node):
        """Visitior for logical expression (i.e. true and false) which is used in compound conditional"""
        literal_value_false = LiteralValue("Boolean", False)
        literal_value_true = LiteralValue("Boolean", True)
        
        # AND: Right side goes in body if, left side in condition
        # OR: Right side goes in body else, left side in condition 
        left, operator, right = node.children
        
        # First we need to check if this is logical and or a logical or
        # The tehcnical types for these are \.or\. and \.and\. so to simplify things we can use the in keyword
        is_or = "or" in operator.type 
        
        top_if = ModelIf()

        top_if_expr = self.visit(left)
        top_if.expr = top_if_expr
        
        bottom_if_expr = self.visit(right)
        if is_or:
            top_if.orelse = [bottom_if_expr]
            top_if.body = [literal_value_true]
        else:
            top_if.orelse = [literal_value_false]
            top_if.body = [bottom_if_expr]

        return top_if

    def visit_assignment_statement(self, node):
        left, _, right = node.children

        # We need to check if the left side is a multidimensional array,
        # Since tree-sitter incorrectly shows this assignment as a call_expression
        if left.type == "call_expression":
            return self._visit_set(node)

        return Assignment(
            left=self.visit(left),
            right=self.visit(right),
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_literal(self, node) -> LiteralValue:
        """Visitor for literals. Returns a LiteralValue"""
        literal_type = node.type
        literal_value = self.node_helper.get_identifier(node)
        literal_source_ref = self.node_helper.get_source_ref(node)

        if literal_type == "number_literal":
            # Check if this is a real value, or an Integer
            if "e" in literal_value.lower() or "." in literal_value:
                return LiteralValue(
                    value_type="AbstractFloat",
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "real"],
                    source_refs=[literal_source_ref],
                )
            else:
                return LiteralValue(
                    value_type="Integer",
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "integer"],
                    source_refs=[literal_source_ref],
                )

        elif literal_type == "string_literal":
            return LiteralValue(
                value_type="Character",
                value=literal_value,
                source_code_data_type=["Fortran", "Fortran95", "character"],
                source_refs=[literal_source_ref],
            )

        elif literal_type == "boolean_literal":
            return LiteralValue(
                value_type="Boolean",
                value=literal_value,
                source_code_data_type=["Fortran", "Fortran95", "logical"],
                source_refs=[literal_source_ref],
            )

        elif literal_type == "array_literal":
            # There are a multiple ways to create an array literal. This visitor is for the traditional explicit creation (/ 1,2,3 /)
            # For the do loop based version, we pass it off to another visitor
            implied_do_loop_expression_node = get_first_child_by_type(
                node, "implied_do_loop_expression"
            )
            if implied_do_loop_expression_node:
                return self._visit_implied_do_loop(implied_do_loop_expression_node)

            return LiteralValue(
                value_type="List",
                value=[
                    self.visit(element) for element in get_non_control_children(node)
                ],
                source_code_data_type=["Fortran", "Fortran95", "dimension"],
                source_refs=[literal_source_ref],
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

    def visit_math_expression(self, node):
        op = self.node_helper.get_identifier(
            get_control_children(node)[0]
        )  # The operator will be the first control character

        operands = []
        for operand in get_non_control_children(node):
            operands.append(self.visit(operand))

        return Operator(
            source_language="Fortran",
            interpreter=None,
            version=None,
            op=op,
            operands=operands,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_variable_declaration(self, node) -> List:
        """Visitor for variable declaration. Will return a List of Var and Assignment nodes."""
        """
        # Node structure
        (variable_declaration)
            (intrinsic_type)
            (type_qualifier)
                (qualifier)
                (value)
            (identifier) ...
            (assignment_statement) ...

        (variable_declaration)
            (derived_type)
                (type_name)
        """
        # A variable can be declared with an intrinsic_type if its built-in, or a derived_type if it is user defined.
        intrinsic_type_node = get_first_child_by_type(node, "intrinsic_type")
        derived_type_node = get_first_child_by_type(node, "derived_type")

        variable_type = ""
        variable_intent = ""

        if intrinsic_type_node:
            type_map = {
                "integer": "Integer",
                "real": "AbstractFloat",
                "complex": None,
                "logical": "Boolean",
                "character": "String",
            }
            variable_type = type_map[self.node_helper.get_identifier(intrinsic_type_node)]
        elif derived_type_node:
            variable_type = self.node_helper.get_identifier(
                get_first_child_by_type(derived_type_node, "type_name", recurse=True),
            )

        # There are multiple type qualifiers that change the way we generate a variable
        # For example, we need to determine if we are creating an array (dimension) or a single variable
        type_qualifiers = get_children_by_types(node, ["type_qualifier"])
        for qualifier in type_qualifiers:
            field = self.node_helper.get_identifier(qualifier.children[0])

            if field == "dimension":
                variable_type = "List"
            elif field == "intent":
                variable_intent = self.node_helper.get_identifier(qualifier.children[1])

        # You can declare multiple variables of the same type in a single statement, so we need to create a Var or Assignment node for each instance
        definied_variables = get_children_by_types(
            node,
            [
                "identifier",  # Variable declaration
                "assignment_statement",  # Variable assignment
                "call_expression",  # Dimension without intent
            ],
        )
        vars = []
        for variable in definied_variables:
            if variable.type == "assignment_statement":
                if variable.children[0].type == "call_expression":
                    vars.append(
                        Assignment(
                            left=self.visit(
                                get_first_child_by_type(
                                    variable.children[0], "identifier"
                                )
                            ),
                            right=self.visit(variable.children[2]),
                            source_refs=[
                                self.node_helper.get_source_ref(variable)
                            ],
                        )
                    )
                    vars[-1].left.type = "dimension"
                    self.variable_context.update_type(
                        vars[-1].left.val.name, "dimension"
                    )
                else:
                    # If its a regular assignment, we can update the type normally
                    vars.append(self.visit(variable))
                    vars[-1].left.type = variable_type
                    self.variable_context.update_type(
                        vars[-1].left.val.name, variable_type
                    )

            elif variable.type == "identifier":
                # A basic variable declaration, we visit the identifier and then update the type
                vars.append(self.visit(variable))
                vars[-1].type = variable_type
                self.variable_context.update_type(vars[-1].val.name, variable_type)
            elif variable.type == "call_expression":
                # Declaring a dimension variable using the x(1:5) format. It will look like a call expression in tree-sitter.
                # We treat it like an identifier by visiting its identifier node. Then the type gets overridden by "dimension"
                vars.append(self.visit(get_first_child_by_type(variable, "identifier")))
                vars[-1].type = "dimension"
                self.variable_context.update_type(vars[-1].val.name, "dimension")

        # By default, all variables are added to a function's list of return values
        # If the intent is actually in, then we need to remove them from the list
        if variable_intent == "in":
            for var in vars:
                self.variable_context.remove_return_value(var.val.name)

        return vars

    def visit_extent_specifier(self, node):
        # Node structure
        # (extent_specifier)
        #   (identifier)
        #   (identifier)

        # The extent specifier is the same as a slice, it can have a start, stop, and step
        # We can determine these by looking at the number of control characters in this node.
        # Fortran uses the character ':' to differentiate these values
        argument_pointer = 0
        arguments = [
            LiteralValue("None", "None"),
            LiteralValue("None", "None"),
            LiteralValue("None", "None"),
        ]
        for child in node.children:
            if child.type == ":":
                argument_pointer += 1
            else:
                arguments[argument_pointer] = self.visit(child)

        return Call(
            func=self.get_gromet_function_node("slice"),
            source_language="Fortran",
            source_language_version="Fortran95",
            arguments=arguments,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_derived_type(self, node: Node) -> RecordDef:
        """Visitor function for derived type definition. Will return a RecordDef"""
        """Node Structure:
        (derived_type_definition)
            (derived_type_statement)
                (base)
                    (base_type_specifier)
                        (identifier)
                (type_name)
            (BODY_NODES)
            ...
        """

        record_name = self.node_helper.get_identifier(
            get_first_child_by_type(node, "type_name", recurse=True)
        )

        # There is no multiple inheritance in Fortran, so a type may only extend 1 other type
        bases = []
        derived_type_statement_node = get_first_child_by_type(
            node, "derived_type_statement"
        )
        base_node = get_first_child_by_type(
            derived_type_statement_node, "identifier", recurse=True
        )
        if base_node:
            bases.append([self.node_helper.get_identifier(base_node)])

        # A derived type can contain symbols with the same name as those already in the main program body.
        # If we tell the variable context we are in a record definition, it will append the type name as a prefix to all defined variables.
        self.variable_context.enter_record_definition(record_name)

        # TODO: Full support for this requires handling the contains statement generally
        funcs = []
        derived_type_procedures_node = get_first_child_by_type(
            node, "derived_type_procedures"
        )
        if derived_type_procedures_node:
            for procedure_node in get_children_by_types(
                derived_type_procedures_node, ["procedure_statement"]
            ):
                funcs.append(
                    self.visit_name(
                        get_first_child_by_type(procedure_node, "method_name")
                    )
                )

        # A derived type can only have variable declarations in its body.
        fields = []
        variable_declarations = [
            self.visit(variable_declaration)
            for variable_declaration in get_children_by_types(
                node, ["variable_declaration"]
            )
        ]
        for declaration in variable_declarations:
            # Variable declarations always returns a list of Var or Assignment, even when only one var is being created
            for var in declaration:
                if isinstance(var, Var):
                    fields.append(var)
                elif isinstance(var, Assignment):
                    # Since this is a record definition, an assignment is actually equivalent to setting the default value
                    var.left.default_value = var.right
                    fields.append(var.left)
                # TODO: Handle dimension type (Call type)
                elif isinstance(var, Call):
                    pass
        # Leaving the record definition sets the prefix back to an empty string
        self.variable_context.exit_record_definition()

        return RecordDef(
            name=record_name,
            bases=bases,
            funcs=funcs,
            fields=fields,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def visit_derived_type_member_expression(self, node) -> Attribute:
        """Visitor function for derived type access. Returns an Attribute object"""
        """ Node Structure
        Scalar Access
        (derived_type_member_expression)
            (identifier)
            (type_member)

        Dimensional Access
        (derived_type_member_expression)
            (call_expression)
                (identifier)
                (argument_list)
            (type_member)
        """

        # If we are accessing an attribute of a scalar type, we can simply pull the name node from the variable context.
        # However, if this is a dimensional type, we must convert it to a call to _get.
        call_expression_node = get_first_child_by_type(node, "call_expression")
        if call_expression_node:
            value = self._visit_get(call_expression_node)
        else:
            # TODO: We shouldn't be accessing get_node directly, since it may not exist in the case of an import.
            # Instead, we should visit the identifier node which will add it to the variable context automatically if it doesn't exist.
            value = self.visit(get_first_child_by_type(node, "identifier", recurse=True))

        attr = self.node_helper.get_identifier(
            get_first_child_by_type(node, "type_member", recurse=True)
        )

        return Attribute(
            value=value,
            attr=attr,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    # NOTE: This function starts with _ because it will never be dispatched to directly. There is not a get node in the tree-sitter parse tree.
    # From context, we will determine when we are accessing an element of a List, and call this function,
    def _visit_get(self, node):
        # Node structure
        # (call_expression)
        #  (identifier)
        #  (argument_list)

        identifier_node = node.children[0]
        argument_nodes = get_non_control_children(node.children[1])

        # First argument to get is the List itself. We can get this by passing the identifier to the identifier visitor
        arguments = []
        arguments.append(self.visit(identifier_node))

        # If there are more than one arguments, then this is a multi dimensional array and we need to use an extended slice
        if len(argument_nodes) > 1:
            dimension_list = LiteralValue()
            dimension_list.value_type = "List"
            dimension_list.value = []
            for argument in argument_nodes:
                dimension_list.value.append(self.visit(argument))

            extslice_call = Call()
            extslice_call.func = self.get_gromet_function_node("ext_slice")
            extslice_call.arguments = []
            extslice_call.arguments.append(dimension_list)

            arguments.append(extslice_call)
        else:
            arguments.append(self.visit(argument_nodes[0]))

        return Call(
            func=self.get_gromet_function_node("get"),
            source_language="Fortran",
            source_language_version="Fortran95",
            arguments=arguments,
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def _visit_set(self, node):
        # Node structure
        # (assignment_statement)
        #  (call_expression)
        #  (right side)

        left, _, right = node.children

        # The left side is equivilent to a call gromet "get", so we will first pass the left side to the get visitor
        # Then we can easily convert this to a "set" call by modifying the fields and then appending the right side to the function arguments
        cast_call = self._visit_get(left)
        cast_call.func = self.get_gromet_function_node("set")
        cast_call.arguments.append(self.visit(right))

        return cast_call

    def _visit_while(self, node) -> Loop:
        """Custom visitor for while loop. Returns a Loop object"""
        """
        Node structure
        Do while
        (do_loop_statement)
            (while_statement)
                (parenthesized_expression)
                    (...) ...
            (body) ...
        """
        while_statement_node = get_first_child_by_type(node, "while_statement")

        # The first body node will be the node after the while_statement
        body = []
        body_start_index = 1 + get_first_child_index(node, "while_statement")
        for body_node in node.children[body_start_index:]:
            child_cast = self.visit(body_node)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            elif isinstance(child_cast, AstNode):
                body.append(child_cast)

        # We don't have explicit handling for parenthesized_expression, but the passthrough handler will make sure that we visit the expression correctly.
        expr = self.visit(
            get_first_child_by_type(while_statement_node, "parenthesized_expression")
        )

        return Loop(
            pre=[],  # TODO: Should pre and post contain anything?
            expr=expr,
            body=body,
            post=[],
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def _visit_implied_do_loop(self, node) -> Call:
        """Custom visitor for implied_do_loop array literal. This form gets converted to a call to range"""
        # TODO: This loop_control is the same as the do loop. Can we turn this into one visitor?
        loop_control_node = get_first_child_by_type(
            node, "loop_control_expression", recurse=True
        )
        loop_control_children = get_non_control_children(loop_control_node)
        if len(loop_control_children) == 3:
            itterator, start, stop = [
                self.visit(child) for child in loop_control_children
            ]
            step = LiteralValue("Integer", "1")
        elif len(loop_control_children) == 4:
            itterator, start, stop, step = [
                self.visit(child) for child in loop_control_children
            ]
        else:
            itterator = None
            start = None
            stop = None
            step = None

        return Call(
            func=self.get_gromet_function_node("range"),
            source_language=None,
            source_language_version=None,
            arguments=[start, stop, step],
            source_refs=[self.node_helper.get_source_ref(node)],
        )

    def _visit_passthrough(self, node):
        if len(node.children) == 0:
            return None

        for child in node.children:
            child_cast = self.visit(child)
            if child_cast:
                return child_cast

    def get_gromet_function_node(self, func_name: str) -> Name:
        # Idealy, we would be able to create a dummy node and just call the name visitor.
        # However, tree-sitter does not allow you to create or modify nodes, so we have to recreate the logic here.
        if self.variable_context.is_variable(func_name):
            return self.variable_context.get_node(func_name)

        return self.variable_context.add_variable(func_name, "function", None)
