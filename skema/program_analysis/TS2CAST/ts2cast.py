import json
from tree_sitter import Language, Parser
from test import Module, SourceRef, Assignment, LiteralValue, Var, VarType, Name, Expr, Operator, AstNode, SourceCodeDataType, ModelImport, List
from cast import CAST

# Naming conventions
# --------------------------------------------------------
# CAST nodes: 
# cast_nodetype

# CAST SourceRef:
# cast_sourceref_attachednode

# Tree-Sitter nodes:
# tree_nodetype_type
# tree_nodetype_identifier
# --------------------------------------------------------

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

    # Set up tree sitter parser
    self.parser = Parser() 
    self.parser.set_language(self.tree_sitter_fortran)
    self.source = """
        program test
          integer,dimension(10) :: x = (/1,2,3,4,5,6,7,8,9,10/)
          integer,dimension(10) :: y
          real,dimension(lev0:lev1,lon0-2:lon1+2),intent(in) :: tn, o2, o1, n2, he
        end program test
        """
    self.tree = self.parser.parse(bytes("""
        program test
          integer,dimension(10) :: x = (/1,2,3,4,5,6,7,8,9,10/)
          integer,dimension(10) :: y
          real,dimension(lev0:lev1,lon0-2:lon1+2),intent(in) :: tn, o2, o1, n2, he
        end program test
        """, "utf8")) # TODO: Set up loading from file
    

    # CAST objects
    self.module_name = None
    self.source_file_name = source_file_path
    self.module = Module()
    self.module.body = []

    # Walking data
    self.variable_id = 0

    # Start visiting
    self.visit(self.tree.root_node)

    # Create outer cast wrapping
    out_cast = CAST([self.module], "Fortran")
    print(
                json.dumps(
                    out_cast.to_json_object(), sort_keys=True, indent=None
                )
            )

  def visit(self, node):
   
    # Generate source_ref
    source_ref = self.generate_source_ref(node)

    # Create map between node children and type
    child_map = {child.type: child for child in node.children}
    
    output = None
    node_type = node.type
    if node_type == "program": # TODO: Should module == translation_unit or program?
      self.visit_program(node, child_map, source_ref)
    elif node_type == "use_statement":
       output = self.visit_use_statement(node, child_map, source_ref)
    elif node_type == "assignment_statement":
      output = self.visit_assignment_statement(node, child_map, source_ref)
    elif node_type == "number_literal":
      output = self.visit_number_literal(node, child_map, source_ref)
    elif node_type == "array_literal":
      output= self.visit_array_literal(node, child_map, source_ref)
    elif node_type == "identifier":
       output = self.visit_identifier(node, child_map, source_ref)
    elif node_type == "math_expression":
       output = self.visit_math_expression(node, child_map, source_ref)
    elif node_type == "variable_declaration":
       output = self.visit_variable_declaration(node, child_map, source_ref)
    else:
      # Visit children
      for child in node.children:
        output = self.visit(child) 
        if isinstance(output, AstNode):
           break # TODO: We break out here because ...

    return output

  def visit_program(self, node, child_map, source_ref):
    for child in node.children:
       cast_ast = self.visit(child)
       if cast_ast:
        # There are some instances where we will recieve more than one CAST node back from a visit,
        # such as in the case of importing multiple symbols from a module
        if isinstance(cast_ast, list):
           self.module.body.extend(cast_ast)
        else:
          self.module.body.append(cast_ast)

    self.module.source_refs = [source_ref]

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
          _, child_source_ref, child_identifer = self.get_node_data(child)

          cast_import = ModelImport()
          cast_import.source_refs = [child_source_ref]
          
          cast_import.name = module_name_identifier
          cast_import.alias = import_alias
          cast_import.all = import_all
          cast_import.symbol = child_identifer

          imports.append(cast_import)
      return imports 
  
  def visit_assignment_statement(self, node, child_map, source_ref):
    cast_assignment = Assignment()
    cast_assignment.source_refs = [source_ref]
    
    left, identifier, right = node.children
    cast_assignment.left = self.visit(left)
    cast_assignment.right = self.visit(right)
    
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
      cast_literal.value_type = "Float"
      cast_literal.source_code_data_type = ["Fortran", "Fortran95", "Real"]
    else:
      cast_literal.value_type = "Integer" 
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
  
  def visit_identifier(self, node, child_map, source_ref):
     cast_var = Var()
     cast_var.source_refs = [source_ref]

     # By default, this is unknown, but can be updated by other visitors
     cast_var.type = "Unknown"
     
     # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
     # Fortran does have optional arguments introduced in F90, but these do not specify a default
     cast_var.default_value = None 

     cast_var.val = Name()

     # The name id is incremented after every use to create a unique ID for every
     # instance of a name. In the AnnCAST passes, incremental ids for each name will be
     # created
     cast_var.val.id = self.variable_id
     self.variable_id += 1
     
     cast_var.val.name = self.get_identifier(source_ref)

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
       "logical": "Bool",
       "character": "String", 
    }
    _, _, type_identifier = self.get_node_data(child_map["intrinsic_type"])
    cast_variable_type = type_map[type_identifier]
  
    # We then need to determine if we are creating an array (dimension) or a single variable
    is_dimension = False
    for child in node.children:
       child_type, _, child_identifier = self.get_node_data(child)
       if child_type == "type_qualifier" and "dimension(": #TODO: Update this logic here
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

        # The name id is incremented after every use to create a unique ID for every
        # instance of a name. In the AnnCAST passes, incremental ids for each name will be
        # created
        cast_var.val.id = self.variable_id
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
  
walker = TS2CAST("exp0.f95", "tree-sitter-fortran")


