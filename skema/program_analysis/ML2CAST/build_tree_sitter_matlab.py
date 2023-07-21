#!/usr/bin/env python3

# build the tree-sitter MatLab shared object file

import os.path
import subprocess
from tree_sitter import Language

# Determine the caller path
CALLER_PATH = os.getcwd()

# Determine absolute path of this file
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Move to project directory
os.chdir(PROJECT_PATH)

# Create the tree-sitter grammar 
ret = subprocess.call('tree-sitter generate', shell = True)
if(ret == 0):
    # Build the C shared object
    Language.build_library(
        os.path.join("build", "ts-matlab.so"),
        'tree-sitter-matlab'
    )
