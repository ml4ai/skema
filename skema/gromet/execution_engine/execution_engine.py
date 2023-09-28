import yaml
import argparse
import asyncio
from pathlib import Path
from typing import Any, List, Dict

import torch
from gqlalchemy import Memgraph

from skema.program_analysis.CAST.pythonAST.builtin_map import retrieve_operator
from skema.gromet.execution_engine.execute import execute_primitive
#TODO: Broken import: from skema.rest.workflows import code_snippets_to_pn_amr
from skema.skema_py.server import System
from skema.gromet.execution_engine.query_runner import QueryRunner
from skema.gromet.execution_engine.symbol_table import SymbolTable

class Execute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primitive: str , inputs: List[torch.Tensor]):
        return execute_primitive(primitive, inputs)

    @staticmethod
    def backward(ctx, grad_output):
        pass
execute = Execute.apply

class ExecutionEngine():
    def __init__(self, host: str, port: str , filename: str):
        self.query_runner = QueryRunner(host, port)
        self.symbol_table = SymbolTable()

        self.filename = filename
    def execute(self, module: bool = False, main: bool = False, function: bool = False, function_name: str = None):
        """Run the execution engine at specified scope"""
        if module:
            module_list = self.query_runner.run_query("module", n_or_m="n", filename=self.filename)
            self.visit(module_list[0])
            print(self.symbol_table.symbols)

    def parameter_extraction(self):
        """Run the execution engine and extract initial values for each parameter"""
        
        # Execute the source at the module level
        self.execute(module=True)

        # Extract the initial values from the symbol map
        return self.symbol_table.get_initial_values()

    def visit(self, node):
        node_types = node._labels
        if "Module" in node_types:
            self.visit_module(node)
        if "Expression" in node_types:
            self.visit_expression(node)
        if "Function" in node_types:
            self.visit_function(node)
        if "Opo" in node_types:
            return self.visit_opo(node)
        if "Opi" in node_types:
            return self.visit_opi(node)
        if "Literal" in node_types:
            return self.visit_literal(node)
        if "Primitive" in node_types:
            return self.visit_primitive(node)

    def visit_module(self, node):
        "Visitor for top-level module"
        node_id = node._id
        expressions = self.query_runner.run_query("ordered_expressions", id=node_id)
        for expression in expressions:
            self.visit(expression)

    def visit_expression(self, node):
        node_id = node._id

        # Only the left hand side is directly connected to the expression.
        # (Expression) -> (Opo) -> (Primitive | Literal | Opo)
        left_hand_side = self.query_runner.run_query("assignment_left_hand", id=node_id)
        right_hand_side = self.query_runner.run_query("assignment_right_hand", id=left_hand_side[0]._id)
       
        # The lefthand side represents the Opo of the variable we are assigning to
        # TODO: What if we have multiple assignment x,y = 1,2
        # TODO: Does an expression always correspond to an assingment?
        symbol = self.visit(left_hand_side[0])
        
        # The right hand side can be either a LiteralValue, an Expression, or a Primitive
        # A Literal
        index = {"Primitive":1, "Expression": 1, "Literal": 2} 
        right_hand_side = sorted(right_hand_side, key=lambda node: index[list(node._labels)[0]])
        value = self.visit(right_hand_side[0])
        if not self.symbol_table.get_symbol(symbol):
            self.symbol_table.add_symbol(symbol, value, None)
        else:
            self.symbol_table.update_symbol(symbol, value, None)
    
    def visit_function(self, node):
        """Visitor for :Opi node type"""

        # First, we need to get all opi symbols and their values

        self.symbol_table.push_function_call()
        #self.visit() for node in None
        self.symbol_table.pop_function_call()
        pass

    def visit_opo(self, node):
        "Visitor for :Opo node type"
        return node.name
    
    def visit_opi(self, node):
        """Visitor for :Opi node type"""
        return node.name
    
    def visit_literal(self, node):
        # Convert to Tensor for execution

        # TODO: Update LiteralValue to remove wrapping "" characters 
        value = node.value["value"].strip("\"")
        value_type = node.value["value_type"]

        if value_type == "Integer":
            return torch.tensor(int(value), dtype=torch.int)
        elif value_type == "AbstractFloat":
            return torch.tensor(float(value), dtype=torch.float)
        elif value_type == "Boolean":
            return torch.tensor(value=="True", dtype=torch.bool)
        elif value_type == "List":
            pass
        elif value_type == "Map":
            pass
        
    def visit_primitive(self, node):
        """Visitor for :Primitive node type"""
        node_id = node._id
        results = self.query_runner.run_query("primitive_operands" , id=node_id)
        inputs = [self.visit(result) for result in results]
        primative = retrieve_operator(node.name)
        return execute(primative, inputs)

if __name__ == "__main__":
    engine = ExecutionEngine("localhost", 7687, "literal_test")
    print(engine.parameter_extraction())
    exit()
    parser = argparse.ArgumentParser(description="Parameter Extraction Script")
    parser.add_argument("source_file", type=str, help="Path to the source file")
    parser.add_argument("host", type=str, help="")
    parser.add_argument("port", type=int, help="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--main", action="store_true", help="Extract parameters from the main module")
    group.add_argument("--function", type=str, metavar="function_name", help="Extract parameters from a specific function")

    args = parser.parse_args()
    
    source_path = Path(args.source_file)
    system = {
        "files": [source_path.name],
        "blobs": [source_path.read_text()],
        "system_name": source_path.name,
        "root_path": source_path.name
    }
    print(system)
    amr = asyncio.run(code_snippets_to_pn_amr(System.parse_obj(system)))
    print(amr)
    #query_runner = QueryRunner(args.host, args.port, Path("queries.yaml"))
    #get_defined_constants(query_runner, source_path.name)
    exit()


