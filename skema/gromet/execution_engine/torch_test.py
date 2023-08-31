import yaml
import argparse
from pathlib import Path
from typing import Any, List, Dict

import torch
from gqlalchemy import Memgraph

from skema.program_analysis.PyAST2CAST.builtin_map import retrieve_operator
from skema.gromet.execution_engine.execute import execute_primitive
from skema.rest.workflows import code_snippets_to_pn_amr
from skema.skema_py.server import System

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Extraction Script")
    parser.add_argument("source_file", type=str, help="Path to the source file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--main", action="store_true", help="Extract parameters from the main module")
    group.add_argument("--function", type=str, metavar="function_name", help="Extract parameters from a specific function")

    args = parser.parse_args()
    system = {}
    exit()
    amr = code_snippets_to_pn_amr(System.parse_obj(system))


class QueryRunner():
    def __init__(self, host: str, port: str, queries_path: Path):
        # First set up the queries map
        self.queries_path = queries_path
        self.queries_map = yaml.safe_load(self.queries_path.read_text())

        # Set up memgrpah instance
        self.memgraph = Memgraph(host=host, port=port)
    def run_query(self, query: str, filename: str = None, function: str = None, id: int = None):
        # Check if query is in query map. Currently we return None if its not found
        # TODO: Improve error handling
        if query not in self.queries_map:
            return None
        query = self.queries_map[query] 

        # There are times we will want to limit the scope we are running queries in.
        # This is done be adding clauses to the cypher queries.
        if filename:
            # The filename query can be added directly to the start of a query.
            # The original query will be added to it 
            filename_query = f"""
            MATCH (n)-[r*]->(m)
            WHERE n.filename='{filename}'
            """
            query = filename_query + query
        if function:
            pass
            function_query = f"""
            MATCH (n)-[r*]->(m)
            WHERE n.name='{function}
            """
        if id:
            id_query = f"""
            MATCH (n)-[r]->(m)
            WHERE id(m)={id}
            return n
            """

        # In most cases, we only want the node objects itself. So we will just return a list of nodes.
        results = self.memgraph.execute_and_fetch(query)
        return [result["m"] for result in results]

class Execute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primitive: str , inputs: List[torch.Tensor]):
        return execute_primitive(primitive, inputs)

    @staticmethod
    def backward(ctx, grad_output):
        pass
execute = Execute.apply

class SymbolMap():
    class Symbol():
        def __init__(self):
            self.scope = ""
            self.state = None
            self.history = []
    def __init__(self):
        self.scopes = {}

    def add_symbol(self):
        pass
    def add_scope(self):
        pass
    def is_symbol(self, symbol:str, scope:str) -> bool:
        pass

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

'''INITIAL_QUERY = """
MATCH (n)-[r]->(m) 
WHERE n.name='example_id0'
RETURN n, r, m;
"""
'''

HOST = "localhost"
PORT = 7687
memgraph = Memgraph(host=HOST, port=PORT)
print(get_defined_constants(memgraph, "simple_test"))
