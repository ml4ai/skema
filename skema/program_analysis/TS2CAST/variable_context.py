from skema.program_analysis.CAST2FN.model.cast import (
   Var,
   Name, 
)

class VariableContext(object):
   def __init__(self):
      self.variable_id = 0
      self.iterator_id = 0
      self.stop_condition_id = 0
      self.context = [{}] # Stack of context dictionaries
      self.context_return_values = [set()] # Stack of context return values
      self.all_symbols = {}
   
   def push_context(self):
      self.context.append({})
      self.context_return_values.append(set())

   def pop_context(self):
      context = self.context.pop()

      # Remove symbols from all_symbols variable
      for symbol in context:
         self.all_symbols.pop(symbol)

      self.context_return_values.pop()

   def add_variable(self, symbol: str, type: str, source_refs: list) -> Name:
      cast_name = Name(source_refs=source_refs)
      cast_name.name = symbol
      cast_name.id = self.variable_id

      # Update variable id
      self.variable_id += 1

      # Add the node to the variable context
      self.context[-1][symbol] = {
        "node": cast_name,
        "type": type,
      }

      # Add reference to all_symbols
      self.all_symbols[symbol] = self.context[-1][symbol]

      return cast_name 
   
   def is_variable(self, symbol: str) -> bool:
      return symbol in self.all_symbols
   
   def get_node(self, symbol: str) -> dict:
      return self.all_symbols[symbol]["node"]
   
   def get_type(self, symbol: str) -> str:
      return self.all_symbols[symbol]["type"]
   
   def update_type(self, symbol:str, type: str):
     self.all_symbols[symbol]["type"] = type

   def add_return_value(self, symbol):
      self.context_return_values[-1].add(symbol)
   def remove_return_value(self, symbol):
      self.context_return_values[-1].discard(symbol)

   def generate_iterator(self):
       symbol = f"generated_iter_{self.iterator_id}"
       self.iterator_id += 1

       return self.add_variable(symbol, "iterator", [None])
   def generate_stop_condition(self):
       symbol = f"sc_{self.stop_condition_id}"
       self.stop_condition_id += 1

       return self.add_variable(symbol, "boolean", [None])
       