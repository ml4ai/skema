import json
from tree_sitter import Language, Parser
from test import Module, SourceRef, Assignment, LiteralValue, Var, VarType, Name, Expr, Operator, AstNode
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
        program example
            integer :: x = 2 + 3
            integer :: y
            integer :: z
            integer :: w

            w = 55 - 54 - 53
            y = x + 3
            z = x + y
            a = (2+99)+(1+98)
            b = ((x+y)-(a+b))
            c = a*(b/(((x+y))))
        end program example
        """
    self.tree = self.parser.parse(bytes("""
        program example
            integer :: x = 2 + 3
            integer :: y
            integer :: z
            integer :: w

            w = 55 - 54 - 53
            y = x + 3
            z = x + y
            a = (2+99)+(1+98)
            b = ((x+y)-(a+b))
            c = a*(b/(((x+y))))
        end program example
        """, "utf8")) # TODO: Set up loading from file
    

    # CAST objects
    self.module_name = None
    self.source_file_name = source_file_path
    self.module = Module()
    self.module.body = []

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

    output = None
    node_type = node.type
    #print(node_type)
    if node_type == "program": # TODO: Should module == translation_unit or program?
      self.visit_program(node, source_ref)
    elif node_type == "assignment_statement":
      output = self.visit_assignment_statement(node, source_ref)
    elif node_type == "number_literal":
      output = self.visit_number_literal(node, source_ref)
    elif node_type == "identifier":
       output = self.visit_identifier(node, source_ref)
    elif node_type == "math_expression":
       output = self.visit_math_expression(node, source_ref)
    else:
      # Visit children
      for child in node.children:
        output = self.visit(child) 
        if isinstance(output, AstNode):
           break # TODO: We break out here because ...

    return output

  def visit_program(self, node, source_ref):
    for child in node.children:
       cast_ast = self.visit(child)
       if cast_ast:
        self.module.body.append(cast_ast)

    self.module.source_refs = [source_ref]

  def visit_assignment_statement(self, node, source_ref):
    cast_assignment = Assignment()
    cast_assignment.source_refs = [source_ref]
    
    left, identifier, right = node.children
    cast_assignment.left = self.visit(left)
    cast_assignment.right = self.visit(right)
    
    return cast_assignment
  
  def visit_number_literal(self, node, source_ref):
    cast_literal = LiteralValue()
    cast_literal.source_refs = [source_ref]

    cast_literal.value_type = "Integer" 
    cast_literal.value = int(self.get_identifier(source_ref))
    cast_literal.source_code_data_type = [] #TODO: Fill this out
    
    return cast_literal

  def visit_identifier(self, node, source_ref):
     cast_var = Var()
     cast_var.source_refs = [source_ref]

     cast_var.type = "Number" # TODO: Generalize
     cast_var.default_value = None # TODO: Handle
     
     cast_var.val = Name()
     cast_var.val.id = 0 # TODO: What is this id?
     cast_var.val.name = self.get_identifier(source_ref)
     cast_var.val.source_refs = [source_ref]

     return cast_var
  
  def visit_math_expression(self, node, source_ref):
    #print(node.children)
    cast_op = Operator()
    cast_op.source_refs = [source_ref]

    cast_op.source_language = "Fortran"
    cast_op.interpreter = "TODO"
    cast_op.version = "TODO"

    left, identifier, right = node.children
    cast_op.op = identifier.type # TODO: Better way to access this? 
    
    # TODO: Differentiate based on number of operators
    cast_op.operands = []
    cast_op.operands.append(self.visit(left))
    cast_op.operands.append(self.visit(right))

    return cast_op
  
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


