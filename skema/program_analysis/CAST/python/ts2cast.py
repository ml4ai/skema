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
    CASTLiteralValue,
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
    ScalarType,
    StructureType
)

from skema.program_analysis.CAST.python.node_helper import (
    NodeHelper,
    get_first_child_by_type,
    get_children_by_types,
    get_first_child_index,
    get_last_child_index,
    get_control_children,
    get_non_control_children,
    FOR_LOOP_LEFT_TYPES,
    FOR_LOOP_RIGHT_TYPES,
    WHILE_COND_TYPES,
    COMPREHENSION_OPERATORS
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

        # Generated FNs by comprehensions/lambdas
        self.generated_fns = []

        # Additional variables used in generation
        self.var_count = 0

        # A dictionary used to keep track of aliases that imports use
        # (like import x as y, or from x import y as z)
        # Used to resolve aliasing in imports
        self.aliases = {}

        # Tree walking structures
        self.variable_context = VariableContext()
        self.node_helper = NodeHelper(self.source, self.source_file_name)

        self.tree = parser.parse(bytes(self.source, "utf8"))

        self.out_cast = self.generate_cast()

    def generate_cast(self) -> List[CAST]:
        '''Interface for generating CAST.'''
        module = self.run(self.tree.root_node)
        module.name = self.source_file_name
        return CAST([generate_dummy_source_refs(module)], "Python") 
        
    def run(self, root) -> List[Module]:
        # In python there's generally only one module at the top level
        # I believe then we just need to visit the root, which is a module
        # Which can then contain multiple things (Handled at module visitor)
        return self.visit(root)

    # TODO: node helper for ignoring comments

    def check_alias(self, name):
        """Given a python string that represents a name,
        this function checks to see if that name is an alias
        for a different name, and returns it if it is indeed an alias.
        Otherwise, the original name is returned.
        """
        if name in self.aliases:
            return self.aliases[name]
        else:
            return name

    def visit(self, node: Node):
        # print(f"===Visiting node[{node.type}]===")
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
        elif node.type == "if_statement":
            return self.visit_if_statement(node)
        elif node.type == "comparison_operator":
            return self.visit_comparison_op(node)
        elif node.type == "assignment":
            return self.visit_assignment(node)
        elif node.type == "attribute":
            return self.visit_attribute(node)
        elif node.type == "identifier":
            return self.visit_identifier(node)
        elif node.type == "unary_operator":
            return self.visit_unary_op(node)
        elif node.type == "binary_operator":
            return self.visit_binary_op(node)
        elif node.type in ["integer", "list"]:
            return self.visit_literal(node)
        elif node.type in ["list_pattern", "pattern_list", "tuple_pattern"]:
            return self.visit_pattern(node)
        elif node.type == "list_comprehension":
            return self.visit_list_comprehension(node)
        elif node.type == "dictionary_comprehension":
            return self.visit_dict_comprehension(node)
        elif node.type == "lambda":
            return self.visit_lambda(node)
        elif node.type == "pair":
            return self.visit_pair(node)
        elif node.type == "while_statement":
            return self.visit_while(node)
        elif node.type == "for_statement":
            return self.visit_for(node)
        elif node.type == "import_statement":
            return self.visit_import(node)
        elif node.type == "import_from_statement":
            return self.visit_import_from(node)
        elif node.type == "class_definition":
            return self.visit_class_definition(node)
        elif node.type == "yield":
            return self.visit_yield(node)
        elif node.type == "assert_statement":
            return self.visit_assert(node)
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
            body=self.generated_fns + body,
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

        parameters = get_children_by_types(node, ["parameters"])[0]
        parameters = get_non_control_children(parameters)
        
        # The body of the function is stored in a 'block' type node
        body = get_children_by_types(node, ["block"])[0].children
        

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

        return ModelReturn(value=get_operand_node(ret_cast), source_refs=[ref])

    def visit_call(self, node: Node) -> Call:
        ref = self.node_helper.get_source_ref(node)

        func_cast = self.visit(node.children[0])

        func_name = get_func_name_node(func_cast)        

        arg_list = get_first_child_by_type(node, "argument_list")
        args = get_non_control_children(arg_list)

        func_args = []
        for arg in args:
            cast = get_name_node(self.visit(arg))
            if isinstance(cast, List):
                func_args.extend(cast)
            elif isinstance(cast, AstNode):
                func_args.append(cast)

        if get_name_node(func_cast).name == "range":
            start_step_value = CASTLiteralValue(
                ScalarType.INTEGER, 
                value="1",
                source_code_data_type=["Python", PYTHON_VERSION, str(type(1))],
                source_refs=[ref]
            )
            # Add a step value
            if len(func_args) == 2:
                func_args.append(start_step_value)
            # Add a start and step value
            elif len(func_args) == 1:
                func_args.insert(0, start_step_value)
                func_args.append(start_step_value)

        # Function calls only want the 'Name' part of the 'Var' that the visit returns
        return Call(
            func=func_name, 
            arguments=func_args, 
            source_refs=[ref]
        )

    def visit_comparison_op(self, node: Node):
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

    def visit_if_statement(self, node: Node) -> ModelIf:
        if_condition = self.visit(get_first_child_by_type(node, "comparison_operator"))

        # Get the body of the if true part
        if_true = get_children_by_types(node, ["block"])[0].children

        # Because in tree-sitter the else if, and else aren't nested, but they're 
        # in a flat level order, we need to do some arranging of the pieces
        # in order to get the correct CAST nested structure that we use
        # Visit all the alternatives, generate CAST for each one
        # and then join them all together
        alternatives = get_children_by_types(node, ["elif_clause","else_clause"])

        if_true_cast = []
        for node in if_true: 
            cast = self.visit(node)
            if isinstance(cast, List):
                if_true_cast.extend(cast)
            elif isinstance(cast, AstNode):
                if_true_cast.append(cast)

        # If we have ts nodes in alternatives, then we're guaranteed
        # at least an else at the end of the if-statement construct
        # We generate the cast for the final else statement, and then
        # reverse the rest of the if-elses that we have, so we can 
        # create the CAST correctly
        final_else_cast = [] 
        if len(alternatives) > 0:
            final_else = alternatives.pop() 
            alternatives.reverse()
            final_else_body = get_children_by_types(final_else, ["block"])[0].children
            for node in final_else_body:
                cast = self.visit(node)
                if isinstance(cast, List):
                    final_else_cast.extend(cast)
                elif isinstance(cast, AstNode):
                    final_else_cast.append(cast)
        
        # We go through any additional if-else nodes that we may have,
        # generating their ModelIf CAST and appending the tail of the 
        # overall if-else construct, starting with the else at the very end
        # We do this tail appending so that when we finish generating CAST the
        # resulting ModelIf CAST is in the correct order
        alternatives_cast = None
        for ts_node in alternatives:
            assert ts_node.type == "elif_clause"
            temp_cast = self.visit_if_statement(ts_node)
            if alternatives_cast == None:
                temp_cast.orelse = final_else_cast
            else:
                temp_cast.orelse = [alternatives_cast]
            alternatives_cast = temp_cast

        if alternatives_cast == None:
            if_false_cast = final_else_cast 
        else:
            if_false_cast = [alternatives_cast]

        return ModelIf(
            expr=if_condition, 
            body=if_true_cast, 
            orelse=if_false_cast, 
            source_refs=[self.node_helper.get_source_ref(node)]
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

        left_cast = get_operand_node(self.visit(left))
        right_cast = get_operand_node(self.visit(right))

        return Operator(
            op=op, 
            operands=[left_cast, right_cast], 
            source_refs=[ref]
        )

    def visit_pattern(self, node: Node):
        pattern_cast = []
        for elem in node.children:
            cast = self.visit(elem)
            if isinstance(cast, List):
                pattern_cast.extend(cast)
            elif isinstance(cast, AstNode):
                pattern_cast.append(cast)

        return CASTLiteralValue(value_type=StructureType.TUPLE, value=pattern_cast) 

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
            return CASTLiteralValue(
                value_type=ScalarType.INTEGER,
                value=literal_value,
                source_code_data_type=["Python", PYTHON_VERSION, str(type(1))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "float":
            return CASTLiteralValue(
                value_type=ScalarType.ABSTRACTFLOAT,
                value=literal_value,
                source_code_data_type=["Python", PYTHON_VERSION, str(type(1.0))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "true" or literal_type == "false":
            return CASTLiteralValue(
                value_type=ScalarType.BOOLEAN,
                value="True" if literal_type == "true" else "False",
                source_code_data_type=["Python", PYTHON_VERSION, str(type(True))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "list":
            list_items = []
            for elem in node.children:
                cast = self.visit(elem)
                if isinstance(cast, List):
                    list_items.extend(cast)
                elif isinstance(cast, AstNode):
                    list_items.append(cast)

            return CASTLiteralValue(
                value_type=StructureType.LIST,
                value = list_items,
                source_code_data_type=["Python", PYTHON_VERSION, str(type([0]))],
                source_refs=[literal_source_ref]
            )
        elif literal_type == "tuple":
            tuple_items = []
            for elem in node.children:
                cast = self.visit(cast)
                if isinstance(cast, List):
                    tuple_items.extend(cast)
                elif isinstance(cast, AstNode):
                    tuple_items.append(cast)

            return CASTLiteralValue(
                value_type=StructureType.LIST,
                value = tuple_items,
                source_code_data_type=["Python", PYTHON_VERSION, str(type((0)))],
                source_refs=[literal_source_ref]
            )

    def handle_dotted_name(self, import_stmt) -> ModelImport:
        ref = self.node_helper.get_source_ref(import_stmt)
        name = self.node_helper.get_identifier(import_stmt)
        self.visit(import_stmt)

        return name

    def handle_aliased_import(self, import_stmt) -> ModelImport:
        ref = self.node_helper.get_source_ref(import_stmt)
        dot_name = get_children_by_types(import_stmt,["dotted_name"])[0]
        name = self.handle_dotted_name(dot_name) 
        alias = get_children_by_types(import_stmt, ["identifier"])[0]
        self.visit(alias)

        return (name, self.node_helper.get_identifier(alias)) 

    def visit_import(self, node: Node):
        ref = self.node_helper.get_source_ref(node)
        to_ret = []

        names_list = get_children_by_types(node, ["dotted_name", "aliased_import"])
        for name in names_list:
            if name.type == "dotted_name":
                resolved_name = self.handle_dotted_name(name)
                to_ret.append(ModelImport(name=resolved_name, alias=None, symbol=None, all=False, source_refs=ref))
            elif name.type == "aliased_import":
                resolved_name = self.handle_aliased_import(name)
                self.aliases[resolved_name[1]] = resolved_name[0]
                to_ret.append(ModelImport(name=resolved_name[0], alias=resolved_name[1], symbol=None, all=False, source_refs=ref))

        return to_ret

    def visit_import_from(self, node: Node):
        ref = self.node_helper.get_source_ref(node)
        to_ret = []

        names_list = get_children_by_types(node, ["dotted_name", "aliased_import"])
        wild_card = get_children_by_types(node, ["wildcard_import"])
        module_name = self.node_helper.get_identifier(names_list[0])

        # if "wildcard_import" exists then it'll be in the list
        if len(wild_card) == 1:
            to_ret.append(ModelImport(name=module_name, alias=None, symbol=None, all=True, source_refs=ref))
        else:
            for name in names_list[1:]:
                if name.type == "dotted_name":
                    resolved_name = self.handle_dotted_name(name) 
                    to_ret.append(ModelImport(name=module_name, alias=None, symbol=resolved_name, all=False, source_refs=ref))
                elif name.type == "aliased_import":
                    resolved_name = self.handle_aliased_import(name)
                    self.aliases[resolved_name[1]] = resolved_name[0]
                    to_ret.append(ModelImport(name=module_name, alias=resolved_name[1], symbol=resolved_name[0], all=False, source_refs=ref))
            
        return to_ret

    def visit_attribute(self, node: Node):
        ref = self.node_helper.get_source_ref(node)
        obj,_,attr = node.children
        obj_cast = self.visit(obj)
        attr_cast = self.visit(attr)

        return Attribute(value= get_name_node(obj_cast), attr=get_name_node(attr_cast), source_refs=ref)

    def handle_for_clause(self, node: Node):
        # Given the "for x in seq" clause of a list comprehension
        # we translate it to a CAST for loop, leaving the actual
        # computation of the body node for the main comprehension handler
        assert node.type == "for_in_clause"
        ref = self.node_helper.get_source_ref(node)

        # NOTE: Assumes the left part with the variable is always the 2nd
        # element in the children and the right part with the actual
        # function call is on the 4th (last) element of the children
        left = self.visit(node.children[1])
        right = self.visit(node.children[-1])

        iterator_name = self.variable_context.generate_iterator()
        stop_cond_name = self.variable_context.generate_stop_condition()
        iter_func = self.get_gromet_function_node("iter")
        next_func = self.get_gromet_function_node("next")
        
        iter_call = Assignment(
            left = Var(iterator_name, "Iterator"),
            right = Call(
                iter_func,
                arguments=[right]
            )
        )

        next_call = Call(
            next_func,
            arguments=[Var(iterator_name, "Iterator")]
        )

        next_assign = Assignment(
            left=CASTLiteralValue(
                "Tuple",
                [
                    left,
                    Var(iterator_name, "Iterator"),
                    Var(stop_cond_name, "Boolean"),
                ],
                source_code_data_type = ["Python",PYTHON_VERSION,"Tuple"],
                source_refs=ref
            ),
            right=next_call
        )

        loop_pre = []
        loop_pre.append(iter_call)
        loop_pre.append(next_assign)

        loop_expr = Operator(
            source_language="Python", 
            interpreter="Python", 
            version=PYTHON_VERSION, 
            op="ast.Eq", 
            operands=[
                stop_cond_name,
                CASTLiteralValue(
                    ScalarType.BOOLEAN,
                    False,
                    ["Python", PYTHON_VERSION, "boolean"],
                    source_refs=ref,
                )
            ], 
            source_refs=ref
        )

        loop_body = [None, next_assign]

        return Loop(pre=loop_pre, expr=loop_expr, body=loop_body, post=[])

    def handle_if_clause(self, node: Node):
        assert node.type == "if_clause"
        ref = self.node_helper.get_source_ref(node)
        conditional = get_children_by_types(node, WHILE_COND_TYPES)[0]
        cond_cast = self.visit(conditional)
        
        return ModelIf(expr=cond_cast,body=[],orelse=[],source_refs=ref)

    def construct_loop_construct(self, node: Node):
        return []

    def visit_list_comprehension(self, node: Node) -> Call:
        ref = self.node_helper.get_source_ref(node)

        temp_list_name = self.variable_context.add_variable(
            "list__temp_", "Unknown", [ref]
        )

        temp_asg_cast = Assignment(
            left=Var(val=temp_list_name), 
            right=CASTLiteralValue(value=[], value_type=StructureType.LIST),
            source_refs = ref
        )

        append_call = self.get_gromet_function_node("append") 
        computation = get_children_by_types(node, COMPREHENSION_OPERATORS)[0]
        computation_cast = self.visit(computation)

        # IDEA: When we see a for_clause we start a new loop construct, and collect if_clauses 
        # as we see them
        clauses = get_children_by_types(node, ["for_in_clause", "if_clause"])
        loop_start = []
        prev_loop = []
        
        if_start = []
        prev_if = []

        for clause in clauses:
            if clause.type == "for_in_clause":
                new_loop = self.handle_for_clause(clause)
                if loop_start == []:
                    loop_start = new_loop
                    prev_loop = loop_start
                else:
                    if prev_if == []:
                        prev_loop.body[0] = new_loop
                        prev_loop = new_loop
                    else:
                        prev_loop.body[0] = prev_if
                        prev_if.body = [new_loop]
                        prev_loop = new_loop
                        if_start = []
                        prev_if = []
            elif clause.type == "if_clause":
                new_if = self.handle_if_clause(clause)
                if if_start == []:
                    if_start = new_if
                    prev_if = if_start
                else:
                    prev_if.body = [new_if]
                    prev_if = new_if
        
        if prev_if == []:
            prev_loop.body[0] = Call(func=Attribute(temp_list_name, append_call), arguments=[computation_cast], source_refs=ref)
        else:
            prev_loop.body[0] = prev_if
            prev_if.body = [Call(func=Attribute(temp_list_name, append_call), arguments=[computation_cast], source_refs=ref)]

        return_cast = ModelReturn(temp_list_name)

        func_name = self.variable_context.generate_func("%comprehension_list")
        func_def_cast = FunctionDef(name=func_name, func_args=[], body=[temp_asg_cast,loop_start,return_cast], source_refs=ref)
        
        self.generated_fns.append(func_def_cast)

        return Call(func=func_name, arguments=[], source_refs=ref)

    def visit_pair(self, node: Node):
        key = self.visit(node.children[0])
        value = self.visit(node.children[2])

        return key,value

    def visit_dict_comprehension(self, node: Node) -> Call:
        ref = self.node_helper.get_source_ref(node)

        temp_dict_name = self.variable_context.add_variable(
            "dict__temp_", "Unknown", [ref]
        )

        temp_asg_cast = Assignment(
            left=Var(val=temp_dict_name),
            right=CASTLiteralValue(value={}, value_type=StructureType.MAP),
            source_refs = ref
        )

        set_call = self.get_gromet_function_node("_set")
        computation = get_children_by_types(node, COMPREHENSION_OPERATORS)[0]
        computation_cast = self.visit(computation)

        # IDEA: When we see a for_clause we start a new loop construct, and collect if_clauses 
        # as we see them
        clauses = get_children_by_types(node, ["for_in_clause", "if_clause"])
        loop_start = []
        prev_loop = []
        
        if_start = []
        prev_if = []

        for clause in clauses:
            if clause.type == "for_in_clause":
                new_loop = self.handle_for_clause(clause)
                if loop_start == []:
                    loop_start = new_loop
                    prev_loop = loop_start
                else:
                    if prev_if == []:
                        prev_loop.body[0] = new_loop
                        prev_loop = new_loop
                    else:
                        prev_loop.body[0] = prev_if
                        prev_if.body = [new_loop]
                        prev_loop = new_loop
                        if_start = []
                        prev_if = []
            elif clause.type == "if_clause":
                new_if = self.handle_if_clause(clause)
                if if_start == []:
                    if_start = new_if
                    prev_if = if_start
                else:
                    prev_if.body = [new_if]
                    prev_if = new_if
        
        if prev_if == []:
            prev_loop.body[0] = Assignment(left=Var(val=temp_dict_name), right=Call(func=set_call, arguments=[temp_dict_name, computation_cast[0].val, computation_cast[1]], source_refs=ref), source_refs=ref)
        else:
            prev_loop.body[0] = prev_if
            prev_loop = Assignment(left=Var(val=temp_dict_name), right=Call(func=set_call, arguments=[temp_dict_name, computation_cast[0].val, computation_cast[1]], source_refs=ref), source_refs=ref)

        return_cast = ModelReturn(temp_dict_name)

        func_name = self.variable_context.generate_func("%comprehension_dict")
        func_def_cast = FunctionDef(name=func_name, func_args=[], body=[temp_asg_cast,loop_start,return_cast], source_refs=ref)

        self.generated_fns.append(func_def_cast)

        return Call(func=func_name, arguments=[], source_refs=ref)
    

    def visit_lambda(self, node: Node) -> Call:
        # TODO: we have to determine how to grab the variables that are being
        # used in the lambda that aren't part of the lambda's arguments
        ref=self.node_helper.get_source_ref(node)
        params = get_children_by_types(node, ["lambda_parameters"])[0] 
        body = get_children_by_types(node, COMPREHENSION_OPERATORS)[0]

        parameters = []
        for param in params.children:
            cast = self.visit(param)
            if isinstance(cast, list):
                parameters.extend(cast)
            else:
                parameters.append(cast)

        body_cast = self.visit(body)
        func_body = body_cast 

        func_name = self.variable_context.generate_func("%lambda")
        func_def_cast = FunctionDef(name=func_name, func_args=parameters, body=[ModelReturn(value=func_body)], source_refs=ref)

        self.generated_fns.append(func_def_cast)
        
        # Collect all the Name node instances to use as arguments for the lambda call
        args = [par.val if isinstance(par, Var) else par for par in parameters]

        return Call(func=func_name, arguments=args, source_refs=ref)

    def visit_while(self, node: Node) -> Loop:
        ref = self.node_helper.get_source_ref(node)
        
        # Push a variable context since a loop 
        # can create variables that only it can see
        self.variable_context.push_context()

        loop_cond_node = get_children_by_types(node, WHILE_COND_TYPES)[0]
        loop_body_node = get_children_by_types(node, ["block"])[0].children

        loop_cond = self.visit(loop_cond_node)

        loop_body = []
        for node in loop_body_node:
            cast = self.visit(node)
            if isinstance(cast, List):
                loop_body.extend(cast)
            elif isinstance(cast, AstNode):
                loop_body.append(cast)

        self.variable_context.pop_context()

        return Loop(
            pre=[],
            expr=loop_cond,
            body=loop_body,
            post=[],
            source_refs = ref
        )

    def visit_for(self, node: Node) -> Loop:
        ref = self.node_helper.get_source_ref(node)

        # Pre: left, right        
        loop_cond_left = get_children_by_types(node, FOR_LOOP_LEFT_TYPES)[0]
        loop_cond_right = get_children_by_types(node, FOR_LOOP_RIGHT_TYPES)[-1]

        # Construct pre and expr value using left and right as needed
        # need calls to "_Iterator"
        
        self.variable_context.push_context()
        iterator_name = self.variable_context.generate_iterator() 
        stop_cond_name = self.variable_context.generate_stop_condition()
        iter_func = self.get_gromet_function_node("iter")
        next_func = self.get_gromet_function_node("next")

        loop_cond_left_cast = self.visit(loop_cond_left)
        loop_cond_right_cast = self.visit(loop_cond_right)

        loop_pre = []
        loop_pre.append(
            Assignment(
                left = Var(iterator_name, "Iterator"),
                right = Call(
                    iter_func,
                    arguments=[loop_cond_right_cast]
                )
            )
        )

        loop_pre.append(
            Assignment(
                left=CASTLiteralValue(
                    "Tuple",
                    [
                        loop_cond_left_cast,
                        Var(iterator_name, "Iterator"),
                        Var(stop_cond_name, "Boolean"),
                    ],
                    source_code_data_type = ["Python",PYTHON_VERSION,"Tuple"],
                    source_refs=ref
                ),
                right=Call(
                    next_func,
                    arguments=[Var(iterator_name, "Iterator")],
                ),
            )
        )

        loop_expr = Operator(
            source_language="Python", 
            interpreter="Python", 
            version=PYTHON_VERSION, 
            op="ast.Eq", 
            operands=[
                stop_cond_name,
                CASTLiteralValue(
                    ScalarType.BOOLEAN,
                    False,
                    ["Python", PYTHON_VERSION, "boolean"],
                    source_refs=ref,
                )
            ], 
            source_refs=ref
        )

        loop_body_node = get_children_by_types(node, ["block"])[0].children
        loop_body = []
        for node in loop_body_node:
            cast = self.visit(node)
            if isinstance(cast, List):
                loop_body.extend(cast)
            elif isinstance(cast, AstNode):
                loop_body.append(cast)

        # Insert an additional call to 'next' at the end of the loop body,
        # to facilitate looping in GroMEt 
        loop_body.append(
            Assignment(
                left=CASTLiteralValue(
                    "Tuple",
                    [
                        loop_cond_left_cast,
                        Var(iterator_name, "Iterator"),
                        Var(stop_cond_name, "Boolean"),
                    ],
                ),
                right=Call(
                    next_func,
                    arguments=[Var(iterator_name, "Iterator")],
                ),
            )
        )

        self.variable_context.pop_context()
        return Loop(
            pre=loop_pre,
            expr=loop_expr,
            body=loop_body,
            post=[],
            source_refs = ref
        )

    def retrieve_init_func(self, functions: List[FunctionDef]):
        # Given a list of CAST function defs, we
        # attempt to retrieve the CAST function that corresponds to
        # "__init__"
        for func in functions:
            if func.name.name == "__init__":
                return func
        return None

    def retrieve_class_attrs(self, init_func: FunctionDef):
        attrs = []
        for stmt in init_func.body:
            if isinstance(stmt, Assignment):
                if isinstance(stmt.left, Attribute):
                    if stmt.left.value.name == "self":
                        attrs.append(stmt.left.attr)

        return attrs

    def visit_class_definition(self, node):
        class_name_node = get_first_child_by_type(node, "identifier")
        class_cast = self.visit(class_name_node)
    
        function_defs = get_children_by_types(get_children_by_types(node, "block")[0], "function_definition")
        func_defs_cast = []
        for func in function_defs:
            func_cast = self.visit(func)
            if isinstance(func_cast, List):
                func_defs_cast.extend(func_cast)
            else:
                func_defs_cast.append(func_cast)

        init_func = self.retrieve_init_func(func_defs_cast)        
        attributes = self.retrieve_class_attrs(init_func)

        return RecordDef(name=get_name_node(class_cast).name, bases=[], funcs=func_defs_cast, fields=attributes)


    def visit_name(self, node):
        # First, we will check if this name is already defined, and if it is return the name node generated previously
        # NOTE: the call to check_alias is a crucial change, to resolve aliasing
        # need to make sure nothing breaks
        identifier = self.check_alias(self.node_helper.get_identifier(node))
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

    def get_gromet_function_node(self, func_name: str) -> Name:
        # Idealy, we would be able to create a dummy node and just call the name visitor.
        # However, tree-sitter does not allow you to create or modify nodes, so we have to recreate the logic here.
        if self.variable_context.is_variable(func_name):
            return self.variable_context.get_node(func_name)

        return self.variable_context.add_variable(func_name, "function", None)
            
    def visit_yield(self, node):
        source_code_data_type = ["Python", "3.8", "List"]
        ref = self.node_helper.get_source_ref(node)
        return [
            CASTLiteralValue(
                StructureType.LIST,
                "YieldNotImplemented",
                source_code_data_type,
                ref
            )   
        ]

    def visit_assert(self, node):
        source_code_data_type = ["Python", "3.8", "List"]
        ref = self.node_helper.get_source_ref(node)
        return [
            CASTLiteralValue(
                StructureType.LIST,
                "AssertNotImplemented",
                source_code_data_type,
                ref
            )   
        ]


def get_name_node(node):
    # Given a CAST node, if it's type Var, then we extract the name node out of it
    # If it's anything else, then the node just gets returned normally
    cur_node = node
    if isinstance(node, list):
        cur_node = node[0]
    if isinstance(cur_node, Attribute):
        return get_name_node(cur_node.attr)
    if isinstance(cur_node, Var):
        return cur_node.val
    else:
        return cur_node

def get_func_name_node(node):
    # Given a CAST node, we attempt to extract the appropriate name element
    # from it. 
    cur_node = node
    if isinstance(cur_node, Var):
        return cur_node.val
    else:
        return cur_node 

def get_operand_node(node):
    # Given a CAST/list node, we extract the appropriate operand for the operator from it
    cur_node = node
    if isinstance(node, list):
        cur_node = node[0]
    if isinstance(cur_node, Var):
        return cur_node.val
    else:
        return cur_node
