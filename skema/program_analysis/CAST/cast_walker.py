import json
import os.path
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Tree, Language, Parser, Node

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


# Generate a CAST for matlab and visit every node

LANGUAGE_LIBRARY_REL_PATH = 'tree-sitter-matlab' # TODO derive

class CastWalker(object):
    def __init__(self, source_file_path: str):
        # Prepare source with preprocessor
        self.path = Path(source_file_path)
        self.source_file_name = self.path.name
        self.source = preprocess(self.path)

        # Run tree-sitter on preprocessor output to generate parse tree
        parser = Parser()
        parser.set_language(
            Language(
                Path(Path(__file__).parent, LANGUAGE_LIBRARY_REL_PATH),
                "matlab"
            )
        )

        self.tree = parser.parse(bytes(self.source, "utf8"))

        # Walking data
        self.variable_context = VariableContext()
        self.node_helper = NodeHelper(self.source, self.source_file_name)

        # Start visiting
        self.out_cast = self.generate_cast()

