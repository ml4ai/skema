# Clean any files created by building the tree-sitter grammar 

import os.path
import subprocess

 
# Determine the caller path
CALLER_PATH = os.getcwd()

# Determine absolute path of this file
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Move to project directory
os.chdir(PROJECT_PATH)

# clean existing generated grammar files
cmd = 'rm -rf bindings.gyp bindings build src'
subprocess.call(cmd, shell = True)

# Move back to caller directory
os.chdir(CALLER_PATH)

