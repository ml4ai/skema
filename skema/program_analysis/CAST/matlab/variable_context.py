from typing import List, Dict, Set
from skema.program_analysis.CAST2FN.model.cast import (
    Var,
    Name,
)

class VariableContext(object):
    def __init__(self):
        self.context = [{}]  # Stack of context dictionaries
        self.context_return_values = [set()]  # Stack of context return values
        self.all_symbols = {}
        self.record_definitions = {}

        # The prefix is used to handle adding Record types to the variable context.
        # This gives each symbol a unqique name. For example "a" would become "type_name.a"
        # For nested type definitions (derived type in a module), multiple prefixes can be added.
        self.prefix = []

        # Flag neccessary to declare if a function is internal or external
        self.internal = False

        self.variable_id = 0
        self.iterator_id = 0
        self.stop_condition_id = 0

    def push_context(self):
        """Create a new variable context and add it to the stack"""
        
        self.context.append({})
        self.context_return_values.append(set())

    def pop_context(self):
        """Pop the current variable context off of the stack and remove any references to those symbols."""
        
        context = self.context.pop()

        # Remove symbols from all_symbols variable
        for symbol in context:
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
        self.all_symbols[full_symbol_name] = self.context[-1][full_symbol_name]

        return cast_name

    def is_variable(self, symbol: str) -> bool:
        """Check if a symbol exists in any context"""
        return symbol in self.all_symbols

    def get_node(self, symbol: str) -> Dict:
        return self.all_symbols[symbol]["node"]

    def get_type(self, symbol: str) -> str:
        return self.all_symbols[symbol]["type"]
