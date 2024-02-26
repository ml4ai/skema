from typing import List, Dict, Set
from skema.program_analysis.CAST2FN.model.cast import (
    Var,
    Name,
)

class VariableContext(object):
    def __init__(self):
        self.context = [{}]  # Stack of context dictionaries
        self.context_return_values = [set()]  # Stack of context return values
        
        # All symbols will use a seperate naming convention to prevent two scopes using the same symbol name
        # The name will be a dot notation list of scopes i.e. scope1.scope2.symbol
        self.all_symbols_scopes = []
        self.all_symbols = {}
    
        # The prefix is used to handle adding Record types to the variable context.
        # This gives each symbol a unqique name. For example "a" would become "type_name.a"
        # For nested type definitions (derived type in a module), multiple prefixes can be added.
        self.prefix = []
        
        # Flag neccessary to declare if a function is internal or external
        self.internal = False

        self.variable_id = 0
        self.iterator_id = 0
        self.stop_condition_id = 0
        self.function_name_id = 0

    def push_context(self):
        """Create a new variable context and add it to the stack"""
        
        # TODO: Could this add unwanted variables to the context or overwrite existing variables
        # If the internal flag is set, then all new scopes will use the top-level context
        if self.internal:
            return None

        self.context.append({})
        self.context_return_values.append(set())

    def pop_context(self):
        """Pop the current variable context off of the stack and remove any references to those symbols."""
     
        # If the internal flag is set, then all new scopes will use the top-level context
        if self.internal:
            return None
        
        context = self.context.pop()

        # Remove symbols from all_symbols variable
        for symbol in context:
            if isinstance(self.all_symbols[symbol], List):
                self.all_symbols[symbol].pop()
            else:
                self.all_symbols.pop(symbol)

        self.context_return_values.pop()

    def add_variable(self, symbol: str, type: str, source_refs: List) -> Name:
        """Add a variable to the current variable context"""
        # Generate the full symbol name using the prefix
        full_symbol_name = ".".join(self.prefix + [symbol])

        cast_name = Name(source_refs=source_refs)
        cast_name.name = symbol
        cast_name.id = self.variable_id

        # Update variable id
        self.variable_id += 1

        # Add the node to the variable context
        self.context[-1][full_symbol_name] = {
            "node": cast_name,
            "type": type,
        }

        # Add reference to all_symbols
        if full_symbol_name in self.all_symbols:
            if isinstance(self.all_symbols[full_symbol_name], List):
                self.all_symbols[full_symbol_name].append(self.context[-1][full_symbol_name])
            else:
                self.all_symbols[full_symbol_name] = [self.all_symbols[full_symbol_name], self.context[-1][full_symbol_name]]
        else:
            self.all_symbols[full_symbol_name] = self.context[-1][full_symbol_name]
        return cast_name

    def is_variable(self, symbol: str) -> bool:
        """Check if a symbol exists in any context"""
        return symbol in self.all_symbols

    def get_node(self, symbol: str) -> Dict:
        if isinstance(self.all_symbols[symbol], List):
            return self.all_symbols[symbol][-1]["node"]
        
        return self.all_symbols[symbol]["node"]

    def get_type(self, symbol: str) -> str:
        if isinstance(self.all_symbols[symbol], List):
            return self.all_symbols[symbol][-1]["type"]
        
        return self.all_symbols[symbol]["type"]

    def update_type(self, symbol: str, type: str):
        """Update the type associated with a given symbol"""
        # Generate the full symbol name using the prefix
        full_symbol_name = ".".join(self.prefix + [symbol])
        if isinstance(self.all_symbols[full_symbol_name], List):
            self.all_symbols[full_symbol_name][-1]["type"] = type
        else:
            self.all_symbols[full_symbol_name]["type"] = type

    def add_return_value(self, symbol):
        self.context_return_values[-1].add(symbol)

    def remove_return_value(self, symbol):
        self.context_return_values[-1].discard(symbol)

    def generate_iterator(self):
        symbol = f"generated_iter_{self.iterator_id}"
        self.iterator_id += 1

        return self.add_variable(symbol, "iterator", None)

    def generate_stop_condition(self):
        symbol = f"sc_{self.stop_condition_id}"
        self.stop_condition_id += 1

        return self.add_variable(symbol, "boolean", None)

    def generate_func(self, name):
        symbol = f"{name}_{self.function_name_id}"
        self.function_name_id += 1

        return self.add_variable(symbol, "unknown", None)

    def enter_record_definition(self, name: str):
        """Enter a record definition. Updates the prefix to the name of the record"""
        self.prefix.append(name)

    def exit_record_definition(self):
        """Exit a record definition. Resets the prefix to the empty string"""
        self.prefix.pop()

    def set_internal(self):
        '''Set the internal flag, meaning, all '''
        self.internal = True

    def unset_internal(self):
        self.internal = False