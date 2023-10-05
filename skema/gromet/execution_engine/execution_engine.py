import yaml
import argparse
import asyncio
import subprocess

from ast import literal_eval
from pathlib import Path
from typing import Any, List, Dict

import torch
from gqlalchemy import Memgraph

from skema.program_analysis.CAST.pythonAST.builtin_map import retrieve_operator
from skema.program_analysis.single_file_ingester import process_file
from skema.gromet.execution_engine.execute import execute_primitive
# TODO: Broken import: from skema.rest.workflows import code_snippets_to_pn_amr
from skema.skema_py.server import System
from skema.gromet.execution_engine.query_runner import QueryRunner
from skema.gromet.execution_engine.symbol_table import SymbolTable
from skema.utils.fold import dictionary_to_gromet_json, del_nulls
SKEMA_BIN = Path(__file__).resolve().parents[2] / "skema-rs" / "skema" / "src" / "bin"

class Execute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primitive: str, inputs: List[torch.Tensor]):
        return execute_primitive(primitive, inputs)

    @staticmethod
    def backward(ctx, grad_output):
        pass


execute = Execute.apply


class ExecutionEngine:
    def __init__(self, host: str, port: str, source_path: str):
        self.query_runner = QueryRunner(host, port)
        self.symbol_table = SymbolTable()
        self.source_path = source_path
        
        # Filename is source path filename minus the extension
        self.filename = Path(source_path).stem

        # Upload source to Memgraph instance
        self.upload_source()

    def upload_source(self):
        """Ingest source file and upload Gromet to Memgraph"""
        
        # Currently, the Gromet ingester writes the output JSON to the directory where the script is run from.
        # Instead, we want to store it alongside the source so that we can upload it to Memgraph.
        gromet_collection = process_file(self.source_path)
        gromet_name = f"{self.filename}--Gromet-FN-auto.json"
        gromet_path = Path(self.source_path).parent / gromet_name
        gromet_path.write_text(dictionary_to_gromet_json(del_nulls(gromet_collection.to_dict())))

        # The Memgraph database should be cleared before execution.
        # Unexpected nodes/edges can cause errors in services making queries.
        self.query_runner.run_query("reset_state")
    
        # Upload to memgraph
        subprocess.run(
            ["cargo", "run", "--bin", "gromet2graphdb", str(gromet_path)], 
            cwd=SKEMA_BIN
        )

    def execute(
        self,
        module: bool = False,
        main: bool = False,
        function: bool = False,
        function_name: str = None,
    ):
        """Run the execution engine at specified scope"""
        if module:
            module_list = self.query_runner.run_query(
                "module", n_or_m="n", filename=self.filename
            )
            self.visit(module_list[0])

    def parameter_extraction(self):
        """Run the execution engine and extract initial values for each parameter"""

        # Execute the source at the module level
        self.execute(module=True)

        # Extract the initial values from the symbol map
        return self.symbol_table.get_initial_values()

    def visit(self, node):
        print("^^^^^^^^^^^^")
        print(node)
        print(node._id)
        print(node._labels)
              
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

        # Only the left hand side is directly connected to the expression. So, we access the right hand side from the left hand side node
        # (Expression) -> (Opo) -> (Primitive | Literal | Opo)
        left_hand_side = self.query_runner.run_query("assignment_left_hand", id=node_id)
        right_hand_side = self.query_runner.run_query(
            "assignment_right_hand", id=left_hand_side[0]._id
        )

        # The lefthand side represents the Opo of the variable we are assigning to
        # TODO: What if we have multiple assignment x,y = 1,2
        # TODO: Does an expression always correspond to an assingment?
        symbol = self.visit(left_hand_side[0])

        # The right hand side can be either a LiteralValue, an Expression, or a Primitive
        # A Literal
        index = {"Primitive": 1, "Expression": 1, "Literal": 2}
        right_hand_side = sorted(
            right_hand_side, key=lambda node: index[list(node._labels)[0]]
        )
        value = self.visit(right_hand_side[0])
        if not self.symbol_table.get_symbol(symbol):
            self.symbol_table.add_symbol(symbol, value, None)
        else:
            self.symbol_table.update_symbol(symbol, value, None)

    def visit_function(self, node):
        """Visitor for :Opi node type"""
        # TODO: Add support for function calls/definitions
        pass

    def visit_opo(self, node):
        "Visitor for :Opo node type"
        return node.name

    def visit_opi(self, node):
        """Visitor for :Opi node type"""
        node_id = node._id

        # If un-named, we need to get the name from the attached Opo
        if node.name == "un-named":
            return self.visit(
                self.query_runner.run_query("assignment_left_hand", id=node_id)[0]
            )

        return node.name

    def visit_literal(self, node):
        def create_dummy_node(value: Dict):
            """Create a dummy gqlalchemy node so that we can pass a LiteralValue to a visitor."""
            class DummyNode():
                pass
            
            node = DummyNode()
            node._id = -1
            node._labels = ["Literal"]
            node.value = value

            # TODO: Update LiteralValue representation for List types
            node.value["value"] = str(node.value["value"])
            
            return node

        # TODO: Update LiteralValue to remove wrapping "" characters
        value = node.value["value"].strip('"')
        value_type = node.value["value_type"]

        if value_type == "Integer":
            return torch.tensor(int(value), dtype=torch.int)
        elif value_type == "AbstractFloat":
            return torch.tensor(float(value), dtype=torch.float)
        elif value_type == "Complex":
            # TODO - Add support for Complex
            pass 
        elif value_type == "Boolean":
            return torch.tensor(value == "True", dtype=torch.bool)
        elif value_type == "List":
            if value == "test":
                return None
    
            list = literal_eval(value)
            return [self.visit(create_dummy_node(element)) for element in list]
            
        elif value_type == "Map":
            # TODO - Add support for Map
            pass
        elif value_type == "None":
            return None

    def visit_primitive(self, node):
        """Visitor for :Primitive node type"""
        node_id = node._id

        # Some inputs may be symbol names, so we need to access the current value from the symbol map
        inputs = [
            self.visit(input)
            for input in self.query_runner.run_query("primitive_operands", id=node_id)
        ]
        inputs = [
            self.symbol_table.get_symbol(input)["current_value"]
            if isinstance(input, str)
            else input
            for input in inputs
        ]

        primative = retrieve_operator(node.name)
        return execute(primative, inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Extraction Script")
    parser.add_argument("source_path", type=str, help="File path to source to execute")
    parser.add_argument(
        "--host",
        default="localhost",
        type=str,
        help="Host serving the memgraph database",
    )
    parser.add_argument(
        "--port", default=7687, type=int, help="Port serving the megraph database"
    )
    args = parser.parse_args()

    engine = ExecutionEngine(args.host, args.port, args.source_path)
    print(engine.parameter_extraction())

    """TODO: Currently the file already has to be uploaded to memgraph. Add support for uploading the file at runtime."""

    """ TODO: New arguments to add with function execution support
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--main", action="store_true", help="Extract parameters from the main module")
    group.add_argument("--function", type=str, metavar="function_name", help="Extract parameters from a specific function")
    """
