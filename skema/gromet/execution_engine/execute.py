from typing import Any, List, Dict
import importlib
import builtins # Used to call builtin functions

module_imports = {}
module_imports["__builtins__"] = builtins

def execute_primitive(primitive: str, inputs: List[Any]) -> Any:
    # If the module path has no . then we assume it is a builtin function 
    # Builtin functions belong to the __builtins__ module
    module_path = primitive.rsplit(".", 1)
    if len(module_path) == 1: 
        module = "__builtins__"
        primitive = primitive
    else:
        module = module_path[0]
        primitive = module_path[1]
    
    # Next, we check if the primitive can be imported from another installed library like operators
    # or Numpy if it is installed
    if module not in module_imports:
        try:
            module_imports[module] = importlib.import_module(module) 
        except:
            print(f"Could not find module to import for: {primitive}")
            return None
    
    # Finally, we attempt to execute the primitive
    try:
        f = getattr(module_imports[module], primitive)
        return f(*inputs)
    except:
        print(f"Could not execute primitive: {primitive}")
        return None


#print(execute_primitive("operator.add", [10, 20]))
#print(execute_primitive("operator.sub", [1,-2]))
#print(execute_primitive("numpy.array", [[1,2,3]]))
#print(execute_primitive("list", [[1,2,3]]))
#print(module_imports)