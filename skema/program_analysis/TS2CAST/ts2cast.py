import json
import os.path
from typing import Any, Dict, List

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
)


from skema.program_analysis.TS2CAST.variable_context import VariableContext
from skema.program_analysis.TS2CAST.node_helper import get_children_by_types, get_identifier, get_source_ref, get_first_child_by_type, get_control_children, get_non_control_children, fix_continuation_lines
from skema.program_analysis.TS2CAST.util import generate_dummy_source_refs, preprocess

from skema.program_analysis.TS2CAST.build_tree_sitter_fortran import LANGUAGE_LIBRARY_REL_PATH

class TS2CAST(object):
    def __init__(self, source_file_path: str):
        # Initialize tree-sitter
        tree_sitter_fortran_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LANGUAGE_LIBRARY_REL_PATH)
        self.tree_sitter_fortran = Language(tree_sitter_fortran_path, "fortran")

        # We load the source code from a file
        with open(source_file_path, "r") as f:
            self.source = f.read()
        self.source = fix_continuation_lines(self.source)

        # Set up tree sitter parser
        self.parser = Parser()
        self.parser.set_language(self.tree_sitter_fortran)
        self.tree = self.parser.parse(bytes(self.source, "utf8"))

        # CAST objects
        self.module_name = None
        self.source_file_name = source_file_path
        self.module = Module()

        # Walking data
        self.variable_context = VariableContext()

        # Start visiting
        self.run(self.tree.root_node)

        generate_dummy_source_refs(self.module)

        # Create outer cast wrapping
        self.out_cast = CAST([self.module], "Fortran")
        print(
                json.dumps(
                       self. out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )

    def run(self, root):
        self.module.source_refs = [get_source_ref(root, self.source_file_name)]
        self.module.body = []
        for child in root.children:
            child_cast = self.visit(child)
            if isinstance(child_cast, List):
                self.module.body.extend(child_cast)
            else:
                self.module.body.append(child_cast)

    def visit(self, node):
        if node.type == "program":
            return self.visit_program_statement(node)
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
        else:
            return self._visit_passthrough(node)

    def visit_program_statement(self, node):
        program_body = []
        for child in node.children[1:]:
            child_cast = self.visit(child)
            if isinstance(child_cast, List):
                program_body.extend(child_cast)
            else:
                program_body.append(child_cast)
        return program_body

    def visit_name(self, node):
        # Node structure
        # (name)

        # First, we will check if this name is already defined, and if it is return the name node generated previously
        identifier = get_identifier(node, self.source)
        if self.variable_context.is_variable(identifier):
            return self.variable_context.get_node(identifier)

        return self.variable_context.add_variable(
            identifier, "Unknown", [get_source_ref(node, self.source_file_name)]
        )

    def visit_function_def(self, node):
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
        statement_node = get_first_child_by_type(node, "subroutine_statement")
        name_node = get_first_child_by_type(statement_node, "name")
        name = self.visit(name_node)  # Visit the name node to add it to the variable context 

        # If this is a function, check for return type and return value
        intrinsic_type = None
        return_value = None
        if node.type == "function":
            signature_qualifiers = get_children_by_types(statement_node, ["intrinsic_type", "function_result"])
            for qualifier in signature_qualifiers:
                if qualifier.type == "intrinsic_type":
                    intrinsic_type = get_identifier(qualifier, self.source)
                    self.variable_context.add_variable(get_identifier(name_node, self.source), intrinsic_type, None)
                elif qualifier.type == "function_result":
                    return_value = self.visit(qualifier.children[0]) #TODO: UPDATE NODES
                    self.variable_context.add_return_value(return_value.val.name)

        # #TODO: What happens if function doesn't return anything?
        # If this is a function, and there is no explicit results variable, then we will assume the return value is the name of the function
        if not return_value:
            self.variable_context.add_return_value(get_identifier(name_node, self.source))

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
                self.variable_context.add_return_value(get_identifier(parameter, self.source))
                func_args.append(self.visit(parameter))

        # The first child of function will be the function statement, the rest will be body nodes
        body = []
        for body_node in node.children[1:]:
            child_cast = self.visit(body_node)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            else:
                body.append(child_cast)
       
        # After creating the body, we can go back and update the var nodes we created for the arguments
        # We do this by looking for intent,in nodes
        for i, arg in enumerate(func_args):
            #print(func_args)
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
            source_refs=[get_source_ref(node, self.source_file_name)]
        )

    def visit_function_call(self, node):
        # Pull relevent nodes
        if node.type == "subroutine_call":
            function_node = node.children[1] 
            arguments_node = node.children[2]
        elif node.type == "call_expression":
            function_node = node.children[0]
            arguments_node = node.children[1]

        function_identifier = get_identifier(function_node, self.source)

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
            func = Name(function_identifier, -1) #TODO: REFACTOR

        # Add arguments to arguments list
        arguments = []
        for argument in arguments_node.children:
            arguments.append(self.visit(argument))

        return Call(
            func=func,
            source_language="Fortran",
            source_language_version="2008",
            arguments = arguments,
            source_refs=[get_source_ref(node, self.source_file_name)]
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
            #TODO: Fix this case
            value = self.variable_context.get_node(return_values[0])
        elif len(return_values) > 1:
            value = LiteralValue(
                value_type="Tuple",
                value=[Var(
                    val=self.variable_context.get_node(ret),
                    type=self.variable_context.get_type(ret),
                    default_value=None,
                    source_refs=None
                ) for ret in return_values],
                source_code_data_type=None, #TODO: REFACTOR
                source_refs = None
            )
        else:
            value = LiteralValue(
                val=None, 
                type=None,
                source_refs=None
            )

        return ModelReturn(
            value=value,
            source_refs=[get_source_ref(node, self.source_file_name)]
        )

    def visit_use_statement(self, node):
        # (use)
        #   (use)
        #   (module_name)
    
        ## Pull relevent child nodes  
        module_name_node = get_first_child_by_type(node, "module_name")
        module_name = get_identifier(module_name_node, self.source)
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
                source_refs=None
            )
        else:
            imports = []
            for symbol in get_non_control_children(included_items_node):
                symbol_identifier = get_identifier(symbol, self.source)
                symbol_source_refs = [get_source_ref(symbol, self.source_file_name)]
                imports.append(
                    ModelImport(
                        name=module_name,
                        alias=import_alias,
                        all=import_all,
                        symbol=symbol_identifier,
                        source_refs=symbol_source_refs
                    )
                )

            return imports

    def visit_do_loop_statement(self, node):
        # Node structure
        # Do loop
        # (do_loop_statement)
        #   (do) - TODO: Get rid of extraneous nodes like this
        #   (loop_control_expression)
        #     (...) ...
        #   (body) ...
        #
        # Do while
        # (do_loop_statement)
        #   (do)
        #   (while_statement)
        #     (while)
        #     (parenthesized_expression)
        #      (...) ...
        #   (body) ...
        # print(self.variable_context.context)
        loop_type = node.children[1].type #TODO: Implement while loop

        # The body will be the same for both loops, like the function definition, its simply every child node after the first
        # TODO: This may not be the case
        body = []
        for body_node in node.children[2:]:
            child_cast = self.visit(body_node)
            if isinstance(child_cast, List):
                body.extend(child_cast)
            else:
                body.append(child_cast)

        # For the init and expression fields, we first need to determine if we are in a regular "do" or a "do while" loop
        # PRE:
        # _next(_iter(range(start, stop, step)))
        loop_control_node = node.children[1]
        itterator = self.visit(loop_control_node.children[0])
        start = self.visit(loop_control_node.children[1])
        stop = self.visit(loop_control_node.children[2])
        step = None
        if len(loop_control_node.children) == 3:  # No step value
            step = LiteralValue("Integer", "1")
        elif len(loop_control_node.children) == 4:
            step = self.visit(loop_control_node.children[3])

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
            source_refs=[get_source_ref(node, self.source_file_name)]
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

        # First we need to identify if this is a componund conditional
        # We can do this by counting the number of control characters in a relational expression
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
                elseif_expr = self.visit(condition.children[2])
                elseif_body = [self.visit(child) for child in condition.children[4:]]

                prev.orelse = ModelIf(elseif_expr, elseif_body, None)
                prev = prev.orelse

        if else_index != -1:
            else_body = [
                self.visit(child)
                for child in node.children[else_index].children[1:]
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
            orelse=orelse,
        )

    def visit_assignment_statement(self, node):
        left, _, right = node.children

        # We need to check if the left side is a multidimensional array,
        # Since tree-sitter incorrectly shows this assignment as a call_expression
        if left.type == "call_expression":
            return self._visit_set(node)

        return Assignment(
            left=self.visit(left),
            right=self.visit(right),
            source_refs=[get_source_ref(node, self.source_file_name)]
        )

    def visit_literal(self, node):
        literal_type = node.type
        literal_value = get_identifier(node, self.source)
        literal_source_ref = get_source_ref(node, self.source_file_name)

        if literal_type == "number_literal":
            # Check if this is a real value, or an Integer
            if "e" in literal_value.lower() or "." in literal_value:
                return LiteralValue(
                    value_type="AbstractFloat", 
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "real"],
                    source_refs=[literal_source_ref])
            else:
                return LiteralValue(
                    value_type="Integer",
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "integer"],
                    source_refs=[literal_source_ref]
                )

        elif literal_type == "string_literal":
            return LiteralValue(
                    value_type="Character",
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "character"],
                    source_refs=[literal_source_ref]
                )

        elif literal_type == "boolean_literal":
            return LiteralValue(
                    value_type="Boolean",
                    value=literal_value,
                    source_code_data_type=["Fortran", "Fortran95", "logical"],
                    source_refs=[literal_source_ref]
                )

        # TODO: Create logic for array literal creation
        elif literal_type == "array_literal":
            return LiteralValue(
                    value_type="List",
                    value=[self.visit(element) for element in get_non_control_children(node)],
                    source_code_data_type=["Fortran", "Fortran95", "dimension"],
                    source_refs=[literal_source_ref]
                )


    def visit_identifier(self, node):
        # By default, this is unknown, but can be updated by other visitors
        identifier = get_identifier(node, self.source)
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
            source_refs=[get_source_ref(node, self.source_file_name)]
        )

    def visit_math_expression(self, node):
        op = get_identifier(get_control_children(node)[0], self.source)  # The operator will be the first control character

        operands = []
        for operand in get_non_control_children(node):
            operands.append(self.visit(operand))

        return Operator(
            source_language="Fortran",
            interpreter=None,
            version=None,
            op=op,
            operands=operands,
            source_refs=[get_source_ref(node, self.source_file_name)]
        )

    def visit_variable_declaration(self, node):
        # Node structure
        # (variable_declaration)
        #   (intrinsic_type)
        #   (type_qualifier)
        #     (qualifier)
        #     (value)
        #   (identifier) ...
        #   (assignment_statement) ...

        # The type will be determined from the child intrensic_type node
        type_map = {
            "integer": "Integer",
            "real": "AbstractFloat",
            "complex": None,
            "logical": "Boolean",
            "character": "String",
        }
    
        intrinsic_type = type_map[get_identifier(get_first_child_by_type(node, "intrinsic_type"), self.source)]
        variable_intent = ""

        # There are multiple type qualifiers that change the way we generate a variable
        # For example, we need to determine if we are creating an array (dimension) or a single variable
        type_qualifiers = get_children_by_types(node, ["type_qualifier"])
        for qualifier in type_qualifiers:
            field = get_identifier(qualifier.children[0], self.source)
    
            if qualifier == "dimension":
                intrinsic_type = "List"
            elif qualifier == "intent":
                variable_intent = get_identifier(qualifier.children[1], self.source)


        # You can declare multiple variables of the same type in a single statement, so we need to create a Var or Assignment node for each instance
        definied_variables = get_children_by_types(node, [
            "identifier", # Variable declaration 
            "assignment_statement", #Variable assignment
            "call_expression" # Dimension without intent
            ])
        vars = []
        for variable in definied_variables:
            if variable.type == "assignment_statement":
                if variable.children[0].type == "call_expression":
                    vars.append(Assignment(
                        left=self.visit(get_first_child_by_type(variable.children[0], "identifier")),
                        right=self.visit(variable.children[2]),
                        source_refs=[get_source_ref(variable, self.source_file_name)]
                    ))
                    vars[-1].left.type="dimension"
                    self.variable_context.update_type(vars[-1].left.val.name, "dimension")
                else:
                    # If its a regular assignment, we can update the type normally
                    vars.append(self.visit(variable)) 
                    vars[-1].left.type = intrinsic_type
                    self.variable_context.update_type(vars[-1].left.val.name, intrinsic_type)

            elif variable.type == "identifier":
                # A basic variable declaration, we visit the identifier and then update the type
                vars.append(self.visit(variable))
                vars[-1].type = intrinsic_type
                self.variable_context.update_type(vars[-1].val.name, intrinsic_type)
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
            source_refs=[get_source_ref(node,self.source_file_name)]
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
            source_refs=[get_source_ref(node, self.source_file_name)]
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

    def _visit_passthrough(self, node):
        if len(node.children) == 0:
            return []

        return self.visit(node.children[0])

    def get_gromet_function_node(self, func_name: str) -> Name:
        # Idealy, we would be able to create a dummy node and just call the name visitor.
        # However, tree-sitter does not allow you to create or modify nodes, so we have to recreate the logic here.
        if self.variable_context.is_variable(func_name):
            return self.variable_context.get_node(func_name)

        return self.variable_context.add_variable(
            func_name, "function", None
        )
