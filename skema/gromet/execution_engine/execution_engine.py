import yaml
import argparse
import asyncio
from pathlib import Path
from typing import Any, List, Dict

import torch
from gqlalchemy import Memgraph

from skema.program_analysis.CAST.pythonAST.builtin_map import retrieve_operator
from skema.gromet.execution_engine.execute import execute_primitive
from skema.rest.workflows import code_snippets_to_pn_amr
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
        #self.symbol_map

    def visit(self, node):
        node_types = node._labels
        if "Module" in node_types:
            self.visit_module(node)
        if "Expression" in node_types:
            self.visit_expression(node)
        if "Opo" in node_types:
            return self.visit_opo(node)
        if "Literal" in node_types:
            return self.visit_literal(node)
        if "Primtive" in node_types:
            return self.visit_primitive(node)

    def visit_module(self, node):
        "Visitor for top-level module"
        node_id = node._id
        expressions = self.query_runner.run_query("ordered_expressions", id=node_id)
        for expression in expressions:
            self.visit(expression)

    def visit_expression(self, node):
        node_id = node._id
        left_hand_side = self.query_runner.run_query("assignment_left_hand", id=node_id)
        right_hand_side = self.query_runner.run_query("assignment_right_hand", id=node_id)
        
        symbol = self.visit(left_hand_side[0])
        value = self.visit(right_hand_side[0])
        
        if not self.symbol_table.get_symbol(symbol):
            self.symbol_table.add_symbol(symbol, value, None)
            print(self.symbol_table.get_all_symbols())
        else:
            self.symbol_table.update_symbol(symbol, value, None)
    
    def visit_opo(self, node):
        "Visitor for :Opo node type"
        return node.name
    
    def visit_literal(self, node):
        # Convert to Tensor for execution

        # The LiteralValue is stored as a str in the Memgraph database
        # So we have to convert it to tokens to parse
        value_tokens = node.value.split()
        value_type = value_tokens[3].replace("\"", "").replace(",", "")
        value = value_tokens[5].replace("\"", "").replace(",", "")

        if value_type == "Integer":
            return torch.tensor(int(value), dtype=torch.int)
        
    def visit_primitive(self, node):
        """Visitor for :Primitive node type"""
        node_id = node._id
        results = self.query_runner.run_query("primitive_operands" , id=node_id)
        inputs = [self.visit(result) for result in results]
        primative = retrieve_operator(node.name)
        return execute(primative, inputs)


def get_defined_constants(query_runner, file_name: str) -> List:
    def backtrack(node):
        """Backtrack over the graph to determine the original value"""
        node_types = node._labels
        node_id = node._id
        if "Opo" in node_types:
            # If this is the top level Opo, we need to get the Expression first
            query = f"""
            MATCH (n)-[r]->(m)
            WHERE id(m)={node_id} and n:Expression
            return n
            """
            results = list(memgraph.execute_and_fetch(query))
            if len(results) == 1:
                return backtrack(results[0]["n"])
    
        elif "Opi" in node_types:
            # If this is an Opi, we need to pass through to the adjavent Opo
            query = f"""
            MATCH (n)-[r]->(m)
            WHERE id(n)={node_id} and m:Opo
            return m
            """
            results = list(memgraph.execute_and_fetch(query))
            return backtrack(results[0]["m"])
        elif "Expression" in node_types:
            # If this is an Expression, we need to get the relevent expression nodes for execution.
            # This will be the Primative, Literal, and OPO? nodes 
            query = f"""
            MATCH (n)-[r]->(m)
            WHERE id(n)={node_id} and (m:Literal or m:Primitive or m:Opi)
            RETURN m
            """
            results = list(memgraph.execute_and_fetch(query))
            literals = [backtrack(result["m"]) for result in results if "Literal" in result["m"]._labels]
            primatives = [backtrack(result["m"]) for result in results if "Primitive" in result["m"]._labels]
            opis = [backtrack(result["m"]) for result in results if "Opi" in result["m"]._labels]
            arguments = literals + opis
            # An expression will either be a single assignment (x=y) or an operation (x=y+1)
            # We can determine this by counting the number of primitve nodes
            if len(primatives)==0:
                return arguments[0]
            else:
                # We need to return the top level primative
                # TODO: Is this the last one?
                return primatives[-1]
            
        elif "Literal" in node_types:
            # Convert to Tensor for execution

            # The LiteralValue is stored as a str in the Memgraph database
            # So we have to convert it to tokens to parse
            value_tokens = node.value.split()
            value_type = value_tokens[3].replace("\"", "").replace(",", "")
            value = value_tokens[5].replace("\"", "").replace(",", "")

            if value_type == "Integer":
                return torch.tensor(int(value), dtype=torch.int)

        elif "Primitive" in node_types:
            results = query_runner.run_query("primitive_operands" , id=node_id)
            inputs = [backtrack(result) for result in results]
            primative = retrieve_operator(node.name)
            return execute(primative, inputs)
    
    # First we want to get all the named symbol nodes
    query = f"""
    MATCH (n)-[r*]->(m)
    WHERE n.filename='{file_name}'
    MATCH (m:Opo) WHERE NOT m.name = 'un-named'
    RETURN m
    """
   
    results = query_runner.run_query("named_opos" ,filename=file_name)
    return {result.name : backtrack(result) for result in results}

if __name__ == "__main__":
    engine = ExecutionEngine("localhost", 7687, "simple_test")
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


