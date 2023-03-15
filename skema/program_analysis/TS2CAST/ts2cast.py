from tree_sitter import Language, Parser


Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    '\\wsl.localhost\Ubuntu\home\vraymond\tree-sitter-fortran',
    'tree-sitter-python'
  ]
)

PY_LANGUAGE = Language('build/my-languages.so', 'python')
FORTRAN_LANGUAGE = Language('build/my-languages.so', 'fortran')

parser = Parser()
parser.set_language(FORTRAN_LANGUAGE)


tree = parser.parse(bytes("""
        program example
            integer :: x = 2
        end program example
        """, "utf8"))


class TS2CAST(object):
  def __init__(self):
    self.program_name = None

  def visit(self, node):
    node_type = node.type
    
    if node_type == "program_statement":
      self.visit_program_statement(node)

    # Visit children
    for child in node.children:
      self.visit(child)

  def visit_program_statement(self, node):

    pass

  @staticmethod
  def range_to_source_ref(range: str):
    pass




def range_to_source_ref(range: str) -> object: # TODO Change to source ref node
  pass


