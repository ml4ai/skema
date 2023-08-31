
class SymbolTable():
    def __init__(self):
        #self.scope = 
        self.symbols = {}

    def enter_scope(self, scope_name):
        self.symbols.append(scope_name)

    def exit_scope(self):
        self.scope_stack.pop()

    def add_symbol(self, name, value, data_type):
        #current_scope = self.scope_stack[-1]
        self.symbols[name] = {
            "current_value": value,
            "history": [value],
            "data_type": data_type,
            "data_type_history": [data_type],
        }

    def update_symbol(self, name, value, data_type):
        #for scope in reversed(self.scope_stack):
        symbol = self.symbols[name]
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
        for scope, symbols in self.symbols.items():
            initial_values[scope] = {name: symbol["history"][0] for name, symbol in symbols.items()}
        return initial_values
