import json
from typing import Any
from dataclasses import dataclass

from tree_sitter import Language, Parser
from test import Module, SourceRef, Assignment, LiteralValue, Var, VarType, Name, Expr, Operator, AstNode, SourceCodeDataType, ModelImport, List, FunctionDef, Loop, Subscript, Boolean, String, Call, ModelReturn
from cast import CAST
#TODO:
# 1. DONE: Check why function argument types aren't being infered from definitions
# 2. DONE: Update variable logic to pass down type
# 2. DONE: Update logic for var id creation
# 3. Update logic for accessing node and node children
# 5. DONE: Fix extraneous import children
# 6. Add logic for return statement in functions and subroutines
# 7. Implement logic for do unitl loop
# 8. Fix logic for continuation lines and comments
# --------------------------------------------------------

class NodeHelper(object):
   def __init__(self, source_file_name:str, source: str):
      self.source_file_name = source_file_name
      self.source = source
    
   def parse_tree_to_dict(self, node) -> dict:
      node_dict = {
         "type": self.get_node_type(node),
         "source_refs": [self.get_node_source_ref(node)],
         "identifier": self.get_node_identifier(node),
         "original_children_order": [],
         "children": [],
         "comments": [],
         "control": []
      }
      
      for child in node.children:
         child_dict = self.parse_tree_to_dict(child)
         node_dict["original_children_order"].append(child_dict)
         if self.is_comment_node(child):  
            node_dict["comments"].append(child_dict)
         elif self.is_control_character_node(child):
            node_dict["control"].append(child_dict)
         else:
            node_dict["children"].append(child_dict)
   
      return node_dict
   
   def is_comment_node(self, node):
      if node.type == "comment":
         return True
      return False
   
   def is_control_character_node(self, node):
      control_characters = [
         ',',
         '=',
         '(',
         ')',
         ":",
         "::",
         "+",
         "-",
         "*",
         "**",
         "/"
      ]
      return node.type in control_characters
      
   
   def get_node_source_ref(self, node) -> SourceRef:
    row_start, col_start = node.start_point
    row_end, col_end = node.end_point
    return SourceRef(self.source_file_name, col_start, col_end, row_start, row_end)
   
   def get_node_identifier(self, node) -> str:
    source_ref = self.get_node_source_ref(node)
    
    line_num = 0
    column_num = 0
    in_identifier = False
    identifier = ""
    for i, char in enumerate(self.source):
        if line_num == source_ref.row_start and column_num == source_ref.col_start:
            in_identifier = True
        elif line_num == source_ref.row_end and column_num == source_ref.col_end:
            break
        
        if char == "\n":
            line_num += 1
            column_num = 0
        else:
            column_num += 1

        if in_identifier:
            identifier += char
    
    return identifier
   
   def get_node_type(self, node) -> str:
      return node.type

   def get_children_by_type(self, node: dict, node_type: str) -> list:
      children = []

      for child in node["children"]:
         if child["type"] == node_type:
            children.append(child)

      return children
   
class VariableContext(object):
   def __init__(self):
      self.variable_id = 0
      self.context = [{}] # Stack of context dictionaries
      self.all_symbols = {}
   
   def push_context(self):
      self.context.append({})
  
   def pop_context(self):
      context = self.context.pop()

      # Remove symbols from all_symbols variable
      for symbol in context:
         self.all_symbols.pop(symbol)

   def is_variable(self, symbol: str) -> bool:
      return symbol in self.all_symbols
   
   def get_node(self, symbol: str) -> dict:
      return self.all_symbols[symbol]["node"]
   
   def get_type(self, symbol: str) -> str:
      return self.all_symbols[symbol]["type"]
   
   def update_type(self, symbol:str, type: str):
     self.context[-1][symbol]["type"] = type

   def add_return_value(self, symbol):
      self.context[-1]["_return_values"].append(symbol)

   def add_variable(self, symbol: str, type: str, source_refs: list) -> Name:
      cast_name = Name(source_refs=source_refs)
      cast_name.name = symbol
      cast_name.id = self.variable_id

      # Update variable id
      self.variable_id += 1

      # Add the node to the variable context
      self.context[-1][symbol] = {
        "node": cast_name,
        "type": type,
      }

      # Add reference to all_symbols
      self.all_symbols[symbol] = self.context[-1][symbol]

      return cast_name
   
   
      

class TS2CAST(object):
  def __init__(self, source_file_path: str, tree_sitter_fortran_path: str):
    
    # Initialize tree-sitter
    self.tree_sitter_fortran = None
    self.tree_sitter_fortran_path = tree_sitter_fortran_path
    Language.build_library(
      # Store the library in the `build` directory
      'build/my-languages.so',

      # Include one or more languages
      [
        self.tree_sitter_fortran_path
      ]
    )
    self.tree_sitter_fortran = Language('build/my-languages.so', 'fortran')

    # We load the source code from a file
    self.source = None
    with open(source_file_path, "r") as f:
       self.source = f.read()
    
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

    self.node_helper = NodeHelper(source_file_path, self.source)
    self.parse_dict = self.node_helper.parse_tree_to_dict(self.tree.root_node)
    #print(json.dumps(self.parse_dict))
    
    
    # Start visiting
    self.run(self.parse_dict)

    # Create outer cast wrapping
    out_cast = CAST([self.module], "Fortran")
    print(
                json.dumps(
                   out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )
            
  
  def run(self, root: dict):
    self.module.source_refs = root["source_refs"]
    for child in root["children"]:
      self.module.body = TS2CAST.update_field(self.module.body, self.visit(child))

  def visit(self, node: dict):
    match node["type"]:
       case "subroutine" | "function":
          return self.visit_function_def(node)
       case "subroutine_call" | "call_expression":
          return self.visit_function_call(node)
       case "use_statement":
          return self.visit_use_statement(node)
       case "variable_declaration":
          return self.visit_variable_declaration(node)
       case "assignment_statement":
          return self.visit_assignment_statement(node)
       case "identifier":
          return self.visit_identifier(node)
       case "name":
          return self.visit_name(node)
       case "math_expression":
          return self.visit_math_expression(node)
       case "number_literal" | "array_literal" | "string_literal" | "boolean_literal":
          return self.visit_literal(node)
       case "keyword_statement":
          return self.visit_keyword_statement(node)
       case "extent_specifier":
          return self.visit_extent_specifier(node)
       case _:
          return []
  
  def visit_name(self, node):
     # Node structure
     # (name)
     assert len(node["children"]) == 0

     # First, we will check if this name is already defined, and if it is
     if self.variable_context.is_variable(node["identifier"]):
        return self.variable_context.get_node(node["identifier"])

     return self.variable_context.add_variable(node["identifier"], "Unknown", node["source_refs"])

  def visit_function_def(self, node):
    # Node structure
    # (subroutine)
    #   (subroutine_statement)
    #     (subroutine) - 
    #     (name)
    #     (parameters) - Optional
    #   (body_node)...
    
    ## Pull relevent child nodes
    function_statement_node = node["children"][0]
    name_node = function_statement_node["children"][1]
    parameters_nodes = self.node_helper.get_children_by_type(function_statement_node, "parameters")
    assert len(parameters_nodes) <= 1 # Should have one or none parameters nodes
  
    # Create a new variable context
    self.variable_context.push_context()

    cast_func = FunctionDef()
    cast_func.source_refs = node["source_refs"]
    cast_func.func_args = []
    cast_func.name= self.visit(name_node)
    
    # Generating the function arguments by walking the parameters node
    if len(parameters_nodes) > 0:
       for child in parameters_nodes[0]["children"]:
          cast_func.func_args = TS2CAST.update_field(cast_func.func_args, self.visit(child))

    # The first child of function will be the function statement, the rest will be body nodes 
    for child in node["children"][1:]:
       cast_func.body = TS2CAST.update_field(cast_func.body, self.visit(child))

    # After creating the body, we can go back and update the var nodes we created for the arguments
    # We do this by looking for intent,in nodes
    for i, arg in enumerate(cast_func.func_args):
       cast_func.func_args[i].type = self.variable_context.get_type(arg.val.name)
    
    # TODO:
    # After everything is finished, we need to update the return value for any Return nodes
    # We will use either the name of the function, or any variables with intent out or inout

    # Pop variable context off of stack before leaving this scope
    self.variable_context.pop_context()

    return cast_func

  def visit_function_call(self, node):
     # TODO: Can functions be called before they are definied

     # Pull relevent nodes
     match(node["type"]):
        case "subroutine_call":
          function_node = node["children"][1]
          arguments_node = node["children"][2]
        case "call_expression":
           function_node = node["children"][0]
           arguments_node = node["children"][1]
     
     function_identifier = function_node["identifier"]
     
     # Tree-Sitter incorrectly parses mutlidimensional array accesses as function calls
     # We will need to check if this is truly a function call or a subscript
     if self.variable_context.is_variable(function_identifier):
        if self.variable_context.get_type(function_identifier) == "List":
            return self._visit_get(node)  #This overrides the visitor and forces us to visit another


     cast_call = Call()
     cast_call.source_refs = node["source_refs"]

     # TODO: What should get a name node? Instrincit functions? Imported functions?
     if self.variable_context.is_variable(function_identifier):  
      cast_call.func = self.variable_context.get_node(function_identifier)
     else:
      cast_call.func = function_identifier
  
     # Add arguments to arguments list
     for child in arguments_node["children"]:
        cast_call.arguments = TS2CAST.update_field(cast_call.arguments, self.visit(child))

     return cast_call
  
  def visit_keyword_statement(self, node):
     keyword_identifier = node["identifier"]

     # Currently, the only keyword_identifier produced by tree-sitter is Return
     # However, there may be other instances, so for now, we will explicitly check first
     if keyword_identifier.lower() == "return":  
      cast_return = ModelReturn()
      cast_return.source_refs = node["source_refs"]

      # In Fortran the return statement doesn't return a value (there is the obsolete "alternative return")
      # We return None here, but will update this value in the function definition visitor
      cast_return.value = None

      return cast_return
     
     return None 

  def visit_use_statement(self, node):
     ## Pull relevent child nodes
     module_name_node = node["children"][0]
     included_items_nodes = self.node_helper.get_children_by_type(node, "included_items",)

     import_all = len(included_items_nodes) == 0
     import_alias = None # TODO: Look into local-name and use-name fields

     # We need to check if this import is a full import of a module, i.e. use module
     # Or a partial import i.e. use module,only: sub1, sub2
     if import_all:
        cast_import = ModelImport()
        cast_import.source_refs = node["source_refs"]
        cast_import.name = module_name_node["identifier"]
        cast_import.alias = import_alias
        cast_import.all = import_all
        cast_import.symbol = None
        
        return cast_import
     else:
      imports = []
      for child in included_items_nodes[0]["children"]:
          cast_import = ModelImport()
          cast_import.source_refs = child["source_refs"]
          cast_import.name = module_name_node["identifier"]
          cast_import.alias = import_alias
          cast_import.all = import_all
          cast_import.symbol = child["identifier"]

          imports.append(cast_import)
      return imports 
  
  def visit_do_loop_statement(self, node, child_map, source_ref):
     cast_loop = Loop()
     cast_loop.source_refs = [source_ref] 
     cast_loop.init = []
     cast_loop.expr = None
     cast_loop.body = []
     
     # The body will be the same for both loops, like the function definition, its simply every child node after the first
     # TODO: This may not be the case
     for child in node.children[1:]:
       child_cast = self.visit(child)
       if child_cast:
          if isinstance(child_cast, list):
            cast_loop.body.extend(child_cast)
          else:
            cast_loop.body.append(child_cast)

     # For the init and expression fields, we first need to determine if we are in a regular "do" or a "do while" loop
     if "loop_control_expression" in child_map:
        '''
        # We need to determine the start, stop, and step for this statement using the loop_control_expression node
        loop_control_node = child_map["loop_control_expression"]
        loop_range = TS2CAST.get_real_children(loop_control_node)
        print(loop_range)
        for i, child in enumerate(loop_control_node.children):
            child_type, _, child_identifier = self.get_node_data(child)

            # We know we will have an init if = is found in the children
            if child_type == "=":
              cast_assignment = Assignment()
              # TODO: SourceRef
              cast_assignment.left = self.visit(loop_control_node.children[i-1])
              cast_assignment.right = self.visit(loop_control_node.children[i+1])
              cast_loop.init.append(cast_assignment)
        
        # Add itt += step to the end of the body
        cast_assignment = Assignment()
        
        #TODO: SourceRef
        cast_loop.body.append(cast_assignment)
        '''
        pass
     elif "while_statement" in child_map:
        cast_loop.expr = self.visit(child_map["while_statement"])
     
     
     '''
     # We need to determine the start, stop, and step for this statement using the loop_control_expression node
     loop_control_node = TS2CAST.search_children_by_type(node, "loop_control_expression")
     for i, child in enumerate(loop_control_node.children):
        child_type, _, child_identifier = self.get_node_data(child)
        # We know we will have an init if = is found in the children
        if child_type == "=":
           cast_assignment = Assignment()
           cast_assignment.left = self.visit(loop_control_node.children[i-1])
           cast_assignment.right = self.visit(loop_control_node.children[i+1])
        
           cast_loop.init.append(cast_assignment)
           
     # Next, we can create the expression thats checked after each itteration
     # Since this is a traditional do loop, the expression will be while(start<stop){x+=step} 
     cast_assignment = Assignment()
     cast_loop.body.append(cast_assignment) # TODO: Should be put at end of body
    '''
     return cast_loop
        
  def visit_assignment_statement(self, node):
    cast_assignment = Assignment()
    cast_assignment.source_refs = node["source_refs"]
    
    assert len(node["children"]) == 2
    left, right = node["children"]

    # We need to check if the left side is a multidimensional array,
    # Since tree-sitter incorrectly shows this assignment as a call_expression
    if left["type"] == "call_expression":
       return self._visit_set(node)
    
    cast_assignment.left = self.visit(left)
    cast_assignment.right = self.visit(right)
    

    
    return cast_assignment
  
  def visit_literal(self, node):
     literal_type = node["type"]
     literal_value = node["identifier"]

     cast_literal = LiteralValue()
     cast_literal.source_refs = node["source_refs"]
  
     match(literal_type):
        case "number_literal":
          # Check if this is a real value, or an Integer
          if 'e' in literal_value.lower() or '.' in literal_value:
            cast_literal.value_type = "AbstractFloat"
            cast_literal.source_code_data_type = ["Fortran", "Fortran95", "real"]
          else:
            cast_literal.value_type = "Integer"
            cast_literal.source_code_data_type = ["Fortran", "Fortran95", "integer"]
          cast_literal.value = literal_value
        
        case "string_literal":
           cast_literal.value_type = "Character"
           cast_literal.source_code_data_type = ["Fortran", "Fortran95", "character"]
           cast_literal.value = literal_value
        
        case "boolean_literal":
           cast_literal.value_type = "Boolean"
           cast_literal.source_code_data_type = ["Fortran", "Fortran95", "logical"]
           cast_literal.value = literal_value

        case "array_literal":
           cast_literal.value_type = "List"
           cast_literal.source_code_data_type = ["Fortran", "Fortran95", "dimension"]
           cast_literal.value = None

     return cast_literal
     
  def visit_identifier(self, node):
     cast_var = Var()
     cast_var.source_refs = node["source_refs"]

     # By default, this is unknown, but can be updated by other visitors
     if self.variable_context.is_variable(node["identifier"]):  
      cast_var.type = self.variable_context.get_type(node["identifier"])
     else:
      cast_var.type = "Unknown"
     
     # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
     # Fortran does have optional arguments introduced in F90, but these do not specify a default
     cast_var.default_value = None 

     # This is another case where we need to override the visitor to explicitly visit another node
     cast_var.val = self.visit_name(node)
     
     return cast_var
  
  def visit_math_expression(self, node):
    op = node["control"][0] # The operator will be the first control character
  
    cast_op = Operator()
    cast_op.source_refs = node["source_refs"]

    cast_op.source_language = "Fortran"
    cast_op.interpreter = None
    cast_op.version = None

    cast_op.op = op["identifier"]
    
    for child in node["children"]:
      cast_op.operands = TS2CAST.update_field(cast_op.operands, self.visit(child))

    return cast_op
  
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
    # TODO: Expand this type map and move it somewhere else
    type_map = {
       "integer": "Integer",
       "real": "AbstractFloat",
       "complex": None,
       "logical": "Boolean",
       "character": "String", 
    }
    
    intrinsic_type = type_map[node["children"][0]["identifier"]] 
    type_qualifiers = self.node_helper.get_children_by_type(node, "type_qualifier")
    identifiers = self.node_helper.get_children_by_type(node, "identifier")
    assignment_statements = self.node_helper.get_children_by_type(node, "asignment_statement")

    # We then need to determine if we are creating an array (dimension) or a single variable
    for type_qualifier in type_qualifiers:
       qualifier = type_qualifier["children"][0]["identifier"]
       value = type_qualifier["children"][1]["identifier"]
       match(qualifier):
          case "dimension":
            intrinsic_type = "List"
          case "intent":
             if value == "out":
                pass
                #print("OUT")
                #self.variable_context.add_return_value()

    # You can declare multiple variables of the same type in a single statement, so we need to create a Var node for each instance
    vars = [] 
    for identifier in identifiers:
      cast_var = self.visit(identifier)
      cast_var.type = intrinsic_type 
      self.variable_context.update_type(cast_var.val.name, intrinsic_type)
      
      vars.append(cast_var)

    for assignment_statement in assignment_statements:
       cast_assignment = self.visit(assignment_statement)
       cast_assignment.left.type = intrinsic_type
       self.variable_context.update_type(cast_assignment.left.name, intrinsic_type)

       vars.append(cast_assignment)

    return vars

  def visit_extent_specifier(self, node):
     # Node structure
     # (extent_specifier)
     #   (identifier)
     #   (identifier)
     

     # The extent specifier is the same as a slice, it can have a start, stop, and step
     # We can determine these by looking at the number of control characters in this node.
     # Fortran uses the character ':' to differentiate these values
     cast_call = Call()
     cast_call.source_refs = node["source_refs"]
     cast_call.func = self.get_gromet_function_node("slice")
     cast_call.arguments = [LiteralValue("None", "None"), LiteralValue("None", "None"), LiteralValue("None", "None")]
     argument_pointer = 0

     for child in node["original_children_order"]:
        if child["type"] == ":":
           argument_pointer += 1
        else:
           cast_call.arguments[argument_pointer] = self.visit(child)
         
     return cast_call

  # NOTE: This function starts with _ because it will never be dispatched to directly. There is not a get node in the tree-sitter parse tree.
  # From context, we will determine when we are accessing an element of a List, and call this function,
  def _visit_get(self, node):
     # Node structure
     # (call_expression)
     #  (identifier)
     #  (argument_list)
   
     assert(len(node["children"]) == 2) 
     identifier = node["children"][0]
     arguments = node["children"][1]["children"]
     
     # This is a call to the get Gromet function
     cast_call = Call()
     cast_call.source_refs = node["source_refs"]

     # We can generate/get the name node for the "get" function by passing the identifier node to the name visitor
     cast_call.func = self.get_gromet_function_node("get")

    # First argument to get is the List itself. We can get this by passing the identifier to the identifier visitor
     cast_call.arguments = []
     cast_call.arguments.append(self.visit(identifier))

     # If there are more than one arguments, then this is a multi dimensional array and we need to use an extended slice
     if len(arguments) > 1:
        dimension_list = LiteralValue()
        dimension_list.source_refs = node["children"][1]["source_refs"]
        dimension_list.value_type = "List"
        dimension_list.value = []
        for argument in arguments:
          dimension_list.value.append(self.visit(argument))
        
        extslice_call = Call()
        extslice_call.source_refs = node["source_refs"]
        extslice_call.func = self.get_gromet_function_node("ext_slice")
        extslice_call.arguments = []
        extslice_call.arguments.append(dimension_list)
        
        cast_call.arguments.append(extslice_call)
     else:
        cast_call.arguments.append(self.visit(arguments[0]))
   
     return cast_call
     
  
  def _visit_set(self, node):
     # Node structure
     # (assignment_statement)
     #  (call_expression)
     #  (right side)

     assert(len(node["children"])) == 2
     left, right = node["children"]

     # The left side is equivilent to a call gromet "get", so we will first pass the left side to the get visitor
     # Then we can easily convert this to a "set" call by modifying the fields and then appending the right side to the function arguments
     cast_call = self._visit_get(left)  
     cast_call.source_refs = node["source_refs"]   
     cast_call.func = self.get_gromet_function_node("set")
     cast_call.arguments.append(self.visit(right))

     return cast_call
  
  def get_gromet_function_node(self, func_name:str) -> Name:
     node_dict = {
        "identifier": func_name,
        "source_refs": [],
        "children": []
     }
     cast_name = self.visit_name(node_dict)
    
     return cast_name
     

  @staticmethod
  def update_field(field: Any, element: Any) -> list:
    if not field:
       field = []

    if element:
      if isinstance(element, list):
        field.extend(element)
      else:
         field.append(element)
    
    return field
  
walker = TS2CAST("test2.f95", "tree-sitter-fortran")

