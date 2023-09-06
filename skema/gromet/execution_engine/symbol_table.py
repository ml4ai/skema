from typing import Dict, List
class SymbolTable():
    def __init__(self):
        self.symbols = {}

        self.current_scope = self.symbols

        # Function definitions is a mapping between function name and the node id of the function definition node in memgraph.
        # We need this because the function body is only represented in memgraph for the first call to a function.
        self.function_definitions = {}

        # Function stack is needed to support nested function calls.
        # It is a list of call scopes [{}]
        self.function_stack = []
        
        # Function history is a dictionary mapping function names to a list of call scopes
        # {
        #   "func1": [{}]
        # }
        self.function_history = {}
        

    def register_function(self, function_name: str, node_id: int):
        """Register a function definition."""
        self.function_definitions[function_name] = node_id

    def get_function_definition(self, function_name: str):
        return self.function_definitions[function_name]
    
    def push_function_call(self, function_name: str, initial_scope: Dict):
        """Push a new function call onto the function stack and set the current scope."""
        self.function_stack.append(initial_scope)
        self.current_scope = self.function_stack[-1]
        

    def pop_function_call(self, function_name: str):
        """Pop a function call"""
        
        # Pop call and add it to history
        scope = self.function_stack.pop()
        if function_name in self.function_history:
            self.function_history[function_name].append(scope)
        else:
            self.function_history[function_name] = [scope]

        # Update the current_scope pointer
        if len(self.function_stack) > 0:
            self.current_scope = self.function_stack[-1]
        else:
            self.current_scope = self.symbols

        

    def add_symbol(self, name, value, data_type):
        self.current_scope[name] = {
            "current_value": value,
            "history": [value],
            "data_type": data_type,
            "data_type_history": [data_type],
        }

    def update_symbol(self, name, value, data_type):
        #for scope in reversed(self.scope_stack):
        symbol = self.current_scope[name]
        symbol["current_value"] = value
        symbol["history"].append(value)
        if symbol["data_type"] != data_type:
            symbol["data_type"] = data_type
            symbol["data_type_history"].append(data_type)

    def get_symbol(self, name):
        return self.symbols.get(name, None)
        

    def get_all_symbols(self):
        return self.symbols

    def get_scope_stack(self):
        return self.scope_stack

    def get_initial_values(self):
        initial_values = {}
        for symbol, symbol_dict in self.symbols.items():
            initial_values = {symbol: symbol_dict["history"][0] for symbol, symbol_dict in self.symbols.items()}
        return initial_values
