import json
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
         "source_ref": self.get_node_source_ref(node),
         "identifier": self.get_node_identifier(node),
         "children": [],
         "comments": [],
         "control": []
      }
      
      for child in node.children:
         child_dict = self.parse_tree_to_dict(child)
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
         ":"
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
   
class VariableContext(object):
   def __init__(self):
      self.context = []
  


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
    self.module.body = []

    # Walking data
    self.variable_id = 0
    self.variable_context = [{}] # Stack of dictionaries

    self.node_helper = NodeHelper(source_file_path, self.source)
    self.parse_dict = self.node_helper.parse_tree_to_dict(self.tree.root_node)

    '''
    # Start visiting
    self.run(self.tree.root_node)

    # Create outer cast wrapping
    out_cast = CAST([self.module], "Fortran")
    print(
                json.dumps(
                    out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )
    '''
  
  def run(self, root):
    self.module.source_refs = [root["source_ref"]]
    for child in root.children:
      self.module.body.extend(self.visit(child))
      '''
      child_cast = self.visit(child)
      if child_cast:
      # There are some instances where we will recieve more than one CAST node back from a visit,
      # such as in the case of importing multiple symbols from a module
       if isinstance(child_cast, list):
        self.module.body.extend(child_cast)
       else:
        self.module.body.append(child_cast)
      '''
    
  
  def visit(self, node):
    # Generate source_ref
    source_ref = self.generate_source_ref(node)

    # Create map between node children and type
    child_map = {child.type: child for child in node.children}
    
    output = None
    node_type = node.type
    
    if node_type == "subroutine" or node_type == "function":
       output = self.visit_function_def(node, child_map, source_ref)
    elif node_type == "subroutine_call" or node_type == "call_expression":
       output = self.visit_function_call(node, child_map, source_ref)
    elif node_type == "keyword_statement":
       output = self.visit_keyword_statement(node, child_map, source_ref)
    elif node_type == "use_statement":
       output = self.visit_use_statement(node, child_map, source_ref)
    elif node_type == "do_loop_statement":
       output = self.visit_do_loop_statement(node, child_map, source_ref)
    elif node_type == "assignment_statement":
      output = self.visit_assignment_statement(node, child_map, source_ref)
    elif node_type == "number_literal":
      output = self.visit_number_literal(node, child_map, source_ref)
    elif node_type == "array_literal":
      output= self.visit_array_literal(node, child_map, source_ref)
    elif node_type == "string_literal":
      output= self.visit_string_literal(node, child_map, source_ref)
    elif node_type == "boolean_literal":
      output= self.visit_boolean_literal(node, child_map, source_ref)
    elif node_type == "identifier":
       output = self.visit_identifier(node, child_map, source_ref)
    elif node_type == "math_expression" or node_type == "relational_expression":
       output = self.visit_math_expression(node, child_map, source_ref) #TODO: Update name of this function
    elif node_type == "variable_declaration":
       output = self.visit_variable_declaration(node, child_map, source_ref)
    else:
      # Visit children
      for child in node.children:
        output = self.visit(child) 
        if isinstance(output, AstNode):
           break # TODO: We break out here because ...

    return output

  def visit_function_def(self, node, child_map, source_ref):
    # Push new variable context onto context stack
    self.variable_context.append({})

    cast_func = FunctionDef()
    cast_func.source_refs = [source_ref]
    cast_func.body = []
    cast_func.func_args = []

    # Function def will always have a name node, but may not have a parameters node
    name_node = TS2CAST.search_children_by_type(node, "name")
    _, _, name_identifier = self.get_node_data(name_node)
    # Like variables functions will also entries in the variable_context table
    cast_name = Name()
    cast_name.name = name_identifier
    cast_name.id = self.variable_id
   
    self.variable_context[-1][name_identifier] = {"id":self.variable_id, "type":"Function"}
    self.variable_id += 1
    
    cast_func.name = cast_name

    # Generating the function arguments by walking the parameters node
    parameters_node = TS2CAST.search_children_by_type(node, "parameters")
    if parameters_node:
       for child in parameters_node.children:
          child_cast = self.visit(child)
          if child_cast:
            cast_func.func_args.append(child_cast)

    # The first child of function will be the function statement, the rest will be body nodes 
    for child in node.children[1:]:
       child_cast = self.visit(child)
       if child_cast:
          if isinstance(child_cast, list):
            cast_func.body.extend(child_cast)
          else:
            cast_func.body.append(child_cast)

    # After creating the body, we can go back and update the var nodes we created for the arguments
    # We do this by looking for intent,in nodes
    for i, arg in enumerate(cast_func.func_args):
       arg_name = arg.val.name
       cast_func.func_args[i].type = self.variable_context[-1][arg_name]["type"]
    
    # After everything is finished, we need to update the return value for any Return nodes
    # We will use either the name of the function, or any variables with intent out or inout
    

    # Pop variable context off of stack before leaving this scope
    self.variable_context.pop()

    return cast_func

  def visit_function_call(self, node, child_map, source_ref):
     # Tree-Sitter incorrectly parses mutlidimensional array accesses as function calls
     # We will need to check if this is truly a function call or a subscript
     cast_call = Call()
     cast_call.source_refs = [source_ref]
     cast_call.arguments = []

     # For both subroutine and functions, the first child will be the identifier, and the second will be an argument list
     _,_,function_identifier = self.get_node_data(node.children[0])
     cast_call.func = function_identifier

     for context in self.variable_context:
        if function_identifier in context:
           if context[function_identifier]["type"] == "List":
            pass #TODO: Handle multidimensional accesss
        
     for child in node.children[1:]:
        cast_child = self.visit(child)
        if cast_child:
           if isinstance(cast_child, list):
              cast_call.arguments.extend(cast_child)
           else:
              cast_call.arguments.append(cast_child)

     return cast_call
  
  def visit_keyword_statement(self, node, child_map, source_ref):
     _,_,keyword_identifer = self.get_node_data(node)
     # Currently, the only keyword_identifier produced by tree-sitter is Return
     # However, there may be other instances, so for now, we will explicitly check first
     if keyword_identifer.lower() == "return":  
      cast_return = ModelReturn()
      cast_return.source_refs = [source_ref]

      # In Fortran the return statement doesn't return a value (there is the obsolete "alternative return")
      # We return None here, but will update this value in the function definition visitor
      cast_return.value = None

      return cast_return
     
     return None 

  def visit_use_statement(self, node, child_map, source_ref):
     # Getting the module information
     _, module_name_source_ref, module_name_identifier = self.get_node_data(child_map["module_name"])

     import_all = "included_items" not in child_map
     import_alias = None # TODO: Look into local-name and use-name fields

     # There are some children in the included_items node that are not actually imports
     # This is a map keeping track of those
     fake_import = {",", "only", ":"}

     # We need to check if this import is a full import of a module, i.e. use module
     # Or a partial import i.e. use module,only: sub1, sub2
     if import_all:
        cast_import = ModelImport()
        cast_import.source_refs = [source_ref] # TODO: What source ref should this be
        
        cast_import.name = module_name_identifier
        cast_import.alias = import_alias
        cast_import.all = import_all
        cast_import.symbol = None
        
        return cast_import
     else:
      imports = []
      for child in child_map["included_items"].children:
          child_type, child_source_ref, child_identifer = self.get_node_data(child)
          if child_type != "identifier":
             continue
          
          cast_import = ModelImport()
          cast_import.source_refs = [child_source_ref]
          
          cast_import.name = module_name_identifier
          cast_import.alias = import_alias
          cast_import.all = import_all
          cast_import.symbol = child_identifer

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
        
  def visit_assignment_statement(self, node, child_map, source_ref):
    cast_assignment = Assignment()
    cast_assignment.source_refs = [source_ref]
    
    left, identifier, right = node.children
    cast_assignment.left = self.visit(left)
    cast_assignment.right = self.visit(right)
    
    # We need to check if the left side is a multidimensional array,
    # Since tree-sitter incorrectly shows this assignment as a call_expression
    if left.type == "call_expression":
       pass
    
    return cast_assignment
  
  def visit_number_literal(self, node, child_map, source_ref):
    
    cast_literal = LiteralValue()
    cast_literal.source_refs = [source_ref]

    # In Fortran, both integer and real correspond to the number_literal node
    # We must check which type of LiteralValue we need to create
    # TODO: Update this code in variable_declaration 
    number_literal = self.get_identifier(source_ref)
    cast_literal.value = number_literal
    if "e" in number_literal.lower() or "." in number_literal.lower():
      cast_literal.value_type = None #TODO: WHAT SHOULD THESE BE
      cast_literal.source_code_data_type = ["Fortran", "Fortran95", "Real"]
    else:
      cast_literal.value_type = None #TODO
      cast_literal.source_code_data_type = ["Fortran", "Fortran95", "Integer"]
    
    return cast_literal
  
  def visit_array_literal(self, node, child_map, source_ref):
     cast_list = List()
     cast_list.source_refs = [source_ref]

     cast_list.values = []
     for child in node.children:
        _, child_source_ref, child_identifier = self.get_node_data(child)
        # The array_literal node may have extraneous children (i.e. , \)
        # We need to check if visiting the child produces a CAST node before appending it
        cast_child = self.visit(child)
        if(cast_child): 
          cast_list.values.append(cast_child)
     
     return cast_list
  
  def visit_string_literal(self, node, child_map, source_ref):
     cast_string = String()
     cast_string.source_refs = [source_ref]

     _,_,string_identifier = self.get_node_data(node)

     cast_string.string = string_identifier

     return cast_string
  
  def visit_boolean_literal(self, node, child_map, source_ref):
     cast_bool = Boolean()
     cast_bool.source_refs = [source_ref]

     _,_,bool_identifier = self.get_node_data(node)

     if bool_identifier == ".true.":
        cast_bool.boolean = True
     else:
        cast_bool.boolean = False

     return cast_bool
     

  def visit_identifier(self, node, child_map, source_ref):
     cast_var = Var()
     cast_var.source_refs = [source_ref]

     # By default, this is unknown, but can be updated by other visitors
     cast_var.type = "Unknown"
     
     # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
     # Fortran does have optional arguments introduced in F90, but these do not specify a default
     cast_var.default_value = None 

     cast_var.val = Name()
     cast_var.val.source_refs = [source_ref]

     cast_var.val.name = self.get_identifier(source_ref)
     
     # TODO: Move this to a function
     # TODO: VARIABLE CONTEXT CODE
     # The name id is incremented after every use to create a unique ID for every
     # instance of a name. In the AnnCAST passes, incremental ids for each name will be
     # created
     found = False
     for context in self.variable_context:
        if cast_var.val.name in context:
           cast_var.val.id = context[cast_var.val.name]["id"]
           cast_var.type = context[cast_var.val.name]["type"]
           found = True
     if not found:
      cast_var.val.id = self.variable_id
      self.variable_context[-1][cast_var.val.name] = {"id": self.variable_id, "type": None}
      self.variable_id += 1
     
     return cast_var
  
  def visit_math_expression(self, node, child_map, source_ref):
    cast_op = Operator()
    cast_op.source_refs = [source_ref]

    cast_op.source_language = "Fortran"
    cast_op.interpreter = "TODO"
    cast_op.version = "Fortran95"

    left, identifier, right = node.children
    cast_op.op = identifier.type
    
    # TODO: Differentiate based on number of operators
    cast_op.operands = []
    cast_op.operands.append(self.visit(left))
    cast_op.operands.append(self.visit(right))

    return cast_op
  
  def visit_variable_declaration(self, node, child_map, source_ref):
    # The type will be determined from the child intrensic_type node
    # TODO: Expand this type map and move it somewhere else
    type_map = {
       "integer": "Number",
       "real": "Number",
       "complex": "Number",
       "logical": "Boolean",
       "character": "String", 
    }
    _, _, type_identifier = self.get_node_data(child_map["intrinsic_type"])
    cast_variable_type = type_map[type_identifier]
  
    # We then need to determine if we are creating an array (dimension) or a single variable
    is_dimension = False
    for child in node.children:
       child_type, _, child_identifier = self.get_node_data(child)
       if child_type == "type_qualifier" and "dimension(" in child_identifier: #TODO: Update this logic here
          cast_variable_type = "List"

    # You can declare multiple variables of the same type in a single statement, so we need to create a Var node for each instance
    vars = [] 
    for child in node.children:
      child_type, child_source_ref, child_identifier = self.get_node_data(child)
      # First check if this is an declaration and definition, or just a declaration
      # If its a definition, then we return an Assignment, else a Var node
      if child_type == "assignment_statement":
         cast_assignment = self.visit(child)
         # Since we have information about the variable type, we are able to update the left side of this assignment
         cast_assignment.left.type = cast_variable_type

         # TODO: VARIABLE CONTEXT CODE
         # We need to update the variable context with these variables as well
         _,_,name_identifier = self.get_node_data(TS2CAST.search_children_by_type(child, "identifier"))
         for i, context in enumerate(self.variable_context):
            if name_identifier in context:
               self.variable_context[i][name_identifier]["type"] = cast_variable_type

         vars.append(cast_assignment)
      elif child_type == "identifier":
        # If just a declaration, create CAST Var node to represent the variable and attach source_ref
        cast_var = Var()
        cast_var.source_refs = [source_ref]

        cast_var.type = cast_variable_type

        # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
        # Fortran does have optional arguments introduced in F90, but these do not specify a default
        cast_var.default_value = None 
      
        cast_var.val = Name()
        cast_var.val.source_refs = [child_source_ref]

        # The variable name comes from the identifier child node
        cast_var.val.name = child_identifier

        # TODO: Move this to a function
        # TODO: VARIABLE CONTEXT CODE
        # The name id is incremented after every use to create a unique ID for every
        # instance of a name. In the AnnCAST passes, incremental ids for each name will be
        # created
        found = False
        for context in self.variable_context:
            if cast_var.val.name in context:
              cast_var.val.id = context[cast_var.val.name]["id"]
              found = True
              # Update typing information
              context[cast_var.val.name]["type"] = cast_variable_type
        if not found:
          cast_var.val.id = self.variable_id
          self.variable_context[-1][cast_var.val.name] = {"id": self.variable_id, "type": cast_variable_type}
          self.variable_id += 1

        vars.append(cast_var)
      else: 
         continue

    return vars

  def generate_source_ref(self, node):
    row_start, col_start = node.start_point
    row_end, col_end = node.end_point

    return SourceRef(self.source_file_name, col_start, col_end, row_start, row_end)

  def get_identifier(self, source_ref: SourceRef) -> str:
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
    
  def get_node_data(self, node):
     node_type = node.type
     source_ref = self.generate_source_ref(node)
     identifier = self.get_identifier(source_ref)
     
     return node_type, source_ref, identifier
  
  @staticmethod
  def get_children_by_type(node, node_type):
    return [child for child in node.children if child.type==node_type]
  @staticmethod
  def search_children_by_type(node, node_type):
     if node.type == node_type:
        return node
     for child in node.children:
        output = TS2CAST.search_children_by_type(child, node_type)
        if output:
           return output
  @staticmethod
  def is_control_character(node):
     return node.type in set([",","(",")","="])
  @staticmethod
  def get_real_children(node):
     output = []
     for child in node.children:
      if not TS2CAST.is_control_character(child):
         output.append(child)
     return output
  
walker = TS2CAST("test.f95", "tree-sitter-fortran")

