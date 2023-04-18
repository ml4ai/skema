"""
    builtin_map.py reads python_builtins.yaml into a structure
    that we can then query and use
"""
# import yaml
# from yaml.loader import SafeLoader
import os
from pathlib import Path

BUILTINS = {'Functions': ["False","None","True","and","as","assert","async","await","break","class",
"continue","def","del","elif","else","except","finally","for","from","global",
"if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
"try","while","with","yield"],
'Operators': ["abs","aiter","all","any","anext","ascii","bin","bool","breakpoint","bytearray","bytes","callable",
 "chr","classmethod","compile","complex","delattr","dict","dir","divmod","enumerate","eval","exec",
 "filter","float","format","frozenset","getattr","globals","hasattr","hash","help","hex","id","input",
 "int","isinstance","issubclass","iter","len","list","locals","map","max","memoryview","min","next",
 "object","oct","open","ord","pow","print","property","range","repr","reversed","round","set","setattr",
 "slice","sorted","staticmethod","str","sum","super","tuple","type","vars","zip","__import__"], 
'Keywords': {"ast.Add":"operator.add","ast.Sub":"operator.sub","ast.Div":"operator.truediv","ast.FloorDiv":"operator.floordiv","ast.Mod":"operator.mod",
"ast.Pow":"operator.pow","ast.LShift":"operator.lshift","ast.Rshift":"operator.rshift","ast.BitOr":"operator.or_","ast.BitAnd":"operator.and_",
"ast.BitXor":"operator.xor","ast.Eq":"operator.eq","ast.NotEq":"operator.ne","ast.Lt":"operator.lt","ast.LtE":"operator.le","ast.Gt":"operator.gt",
"ast.GtE":"operator.ge","ast.In":"operator.contains","ast.Is":"operator.is","ast.IsNot":"operator.is_not","ast.MatMul":"operator.matmul",
"ast.UAdd":"operator.pos","ast.USub":"operator.neg","ast.Not":"operator.not_","ast.Invert":"operator.invert"}}

#["abs","aiter","all","any","anext","ascii","bin","bool","breakpoint","bytearray","bytes","callable",
# "chr","classmethod","compile","complex","delattr","dict","dir","divmod","enumerate","eval","exec",
#"filter","float","format","frozenset","getattr","globals","hasattr","hash","help","hex","id","input",
# "int","isinstance","issubclass","iter","len","list","locals","map","max","memoryview","min","next",
#"object","oct","open","ord","pow","print","property","range","repr","reversed","round","set","setattr",
#"slice","sorted","staticmethod","str","sum","super","tuple","type","vars","zip","__import__"]

#["False","None","True","and","as","assert","async","await","break","class",
#"continue","def","del","elif","else","except","finally","for","from","global",
#"if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
#"try","while","with","yield"]

def build_map(filename="python_builtins.yaml"):
    global BUILTINS
    skema_root = "skema/skema/program_analysis/PyAST2CAST/"
    if BUILTINS == None: 
        with open(filename) as f:
            # BUILTINS = yaml.load(f, Loader=SafeLoader)
            pass

def dump_map():
    if BUILTINS != None:
        print(BUILTINS)
    else:
        print("Built in map isn't generated yet")

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

    return "NOT_IMPLEMENTED"
        


# Test
def main():
    build_map()

main()