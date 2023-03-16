import json
from tree_sitter import Language, Parser
from test import Module, SourceRef, Assignment, LiteralValue, Var, VarType, Name, Expr, Operator, AstNode, SourceCodeDataType, ModelImport
from cast import CAST

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
          use cons_module,only: rmassinv_o2,rmassinv_o1,rmassinv_n2, rmassinv_he,gask,t0
          use addfld_module,only: addfld
          use diags_module,only: mkdiag_MU_M
          implicit none
        end program test
        """
    self.tree = self.parser.parse(bytes("""
        program test
          use cons_module,only: rmassinv_o2,rmassinv_o1,rmassinv_n2, rmassinv_he,gask,t0
          use addfld_module,only: addfld
          use diags_module,only: mkdiag_MU_M
          implicit none
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
     imports = []
     
     module_name_node = child_map["module_name"]
     module_name_node_source_ref = self.generate_source_ref(module_name_node)
     module_name = self.get_identifier(module_name_node_source_ref)

     included_items_node = child_map["included_items"]

     # We need to check if this import is a full import of a module, i.e. use module
     # Or a partial import i.e. use module,only: sub1, sub2
     
     print(included_items_node.children)
     if "only" in child_map:
        print("HERE")
     else:
        pass
     for child in included_items_node.children:
        child_source_ref = self.generate_source_ref(child)

        cast_import = ModelImport()
        cast_import.source_refs = [child_source_ref]
        
        cast_import.name = module_name
        cast_import.alias = None
        cast_import.symbol = self.get_identifier(child_source_ref)

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

    type_node = child_map["intrinsic_type"]
    type_node_source_ref = self.generate_source_ref(type_node)
    intrinsic_type = self.get_identifier(type_node_source_ref)
    cast_variable_type = type_map[intrinsic_type]
    
    # First check if this is an declaration and definition, or just a declaration
    # If its a definition, then we return an Assignment, else a Var node
    if "assignment_statement" in child_map:
      cast_assignment = self.visit(child_map["assignment_statement"])
      cast_var = cast_assignment.left 
      cast_var.type = cast_variable_type

      return cast_assignment
  
    # If just a declaration, create CAST Var node to represent the variable and attach source_ref
    cast_var = Var()
    cast_var.source_refs = [source_ref]
    
    # Default value comes from Pytohn keyword arguments i.e. def foo(a, b=10)
    # Fortran does have optional arguments introduced in F90, but these do not specify a default
    cast_var.default_value = None 

    cast_var_name_node = child_map["identifier"]
    cast_var_name_source_ref = self.generate_source_ref(cast_var_name_node)
    
    cast_var.val = Name()
    cast_var.val.source_refs = [cast_var_name_source_ref]

    # The variable name comes from the identifier child node
    cast_var.val.name = self.get_identifier(cast_var_name_source_ref)

    # The name id is incremented after every use to create a unique ID for every
    # instance of a name. In the AnnCAST passes, incremental ids for each name will be
    # created
    cast_var.val.id = self.variable_id
    self.variable_id += 1

    return cast_var
  

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
  
walker = TS2CAST("exp0.f95", "tree-sitter-fortran")


