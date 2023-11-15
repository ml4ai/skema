import pytest
import torch
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

from skema.rest.proxies import SKEMA_GRAPH_DB_HOST, SKEMA_GRAPH_DB_PORT
from skema.gromet.execution_engine.execution_engine import ExecutionEngine

MEMGRAPH_CI_HOST = SKEMA_GRAPH_DB_HOST
MEMGRAPH_CI_PORT = int(SKEMA_GRAPH_DB_PORT)

@pytest.mark.ci_only
def test_parameter_extraction():
    """Unit test for testing basic parameter extraction with execution engine"""
    input = """
x = 2
y = x+1
z = x+y
"""
    expected_output = {"x": torch.tensor(2), "y": torch.tensor(3), "z": torch.tensor(5)}

    with TemporaryDirectory() as temp:
        source_path = Path(temp) / "test_parameter_extraction.py"
        source_path.write_text(input)

        output = ExecutionEngine(
            host=MEMGRAPH_CI_HOST, port=MEMGRAPH_CI_PORT, source_path=source_path
        ).parameter_extraction()

        # torch.tensor overrides the equality '==' operator, so the following is a valid check
        assert output == expected_output
