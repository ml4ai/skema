"""
    builtin_map.py reads python_builtins.yaml into a structure
    that we can then query and use
"""
import yaml
from yaml.loader import SafeLoader

import os
from pathlib import Path

BUILTINS_FILENAME = "python_builtins.yaml"
BUILTINS = None

def build_map():
    global BUILTINS
    if BUILTINS == None: 
        f_path = os.path.join(os.path.dirname(__file__), BUILTINS_FILENAME)
        with open(f_path) as f:
            BUILTINS = yaml.load(f, Loader=SafeLoader)
        return True
    else:
        return False

def dump_map():
    if BUILTINS != None:
        print(BUILTINS)
        return True
    else:
        print("Built in map isn't generated yet")
        return False

def check_builtin(func_name):
    # Check if it's in the list of functions
    # Then check the actual operators afterwards

    if func_name in BUILTINS['Functions']:
        return True
    for op in BUILTINS['Operators']:
        if func_name in op:
            return True
    
    return False
    
def retrieve_operator(func_name):
    # Returns the function name if it's a builtin function
    # Otherwise it returns the operator function name if it exists
    # TODO: Vincent double check this functionality

    if func_name in BUILTINS['Functions']:
        return func_name

    for op in BUILTINS['Operators']:
        if func_name in op:
            return op[func_name]

    print(f"{func_name} NOT IMPLEMENTED")
    return None
