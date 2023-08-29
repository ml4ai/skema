import json
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, List, Union

from tree_sitter import Language, Parser, Node, Tree

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

from skema.program_analysis.CAST.matlab.matlab_tree_builder import MATLAB_TREE_BUILDER
from skema.program_analysis.CAST.matlab.variable_context import VariableContext
from skema.program_analysis.CAST.matlab.node_helper import (
    NodeHelper,
    get_children_by_types,
    get_first_child_by_type,
    get_control_children,
    get_non_control_children,
    get_first_child_index,
    get_last_child_index,
)
from skema.program_analysis.CAST.matlab.util import generate_dummy_source_refs
#
#from skema.program_analysis.CAST.matlab.preprocessor.preprocess import preprocess
from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH

#TODO:  Get from grammar
MATLAB_VERSION='matlab_version_here'

class MATLAB_TEST(object):
    def __init__(self):
        
        # get a tree-sitter tree based on source input
        matlab_tree_builder = MATLAB_TREE_BUILDER()

    def test(self, title, str, source: str):

        print('\n\n' + title)
        print('\nSOURCE:')
        print(source)

        print('\nSYNTAX TREE:')
        tree = matlab_tree_builder.get_tree(source)
        matlab_tree_builder.traverse_tree(tree)

        return tree

        

