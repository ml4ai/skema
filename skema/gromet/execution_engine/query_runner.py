import yaml
import traceback
from pathlib import Path

from neo4j import GraphDatabase

QUERIES_PATH = Path(__file__).parent / "queries.yaml"


class QueryRunner:
    def __init__(self, protocol: str, host: str, port: str):
        # First set up the queries map
        self.queries_path = QUERIES_PATH
        self.queries_map = yaml.safe_load(self.queries_path.read_text())

        # Set up memgrpah instance
        self.memgraph =  GraphDatabase.driver(uri=f"{protocol}{host}:{port}", auth=("", ""))
        self.memgraph.verify_connectivity()
    
    def run_query(
        self,
        query: str,
        n_or_m: str = "m",
        filename: str = None,
        function: str = None,
        id: str = None,
    ):
        # Check if query is in query map. Currently we return None if its not found
        # TODO: Improve error handling
        if query not in self.queries_map:
            return None
        query = self.queries_map[query]

        # There are times we will want to limit the scope we are running queries in.
        # This is done be adding clauses to the cypher queries.
        if filename:
            query = query.replace("$FILENAME", filename)

        if id:
            query = query.replace("$ID", str(id))

        # In most cases, we only want the node objects itself. So we will just return a list of nodes.
        records,summary,keys = self.memgraph.execute_query(query, database_="memgraph")
        return neo4j_to_memgprah(records, n_or_m)
  

def neo4j_to_memgprah(neo4j_output, n_or_m: str):
    """Converts neo4j output format to memgraph output format"""
    class DummyNode():
        pass
    
    results = []
    for record in neo4j_output:
        node_ptr = dict(record)[n_or_m]

        dummy_node = DummyNode()
        dummy_node._labels = list(node_ptr.labels)
        dummy_node._id = node_ptr.element_id

        for key, value in node_ptr._properties.items():
            setattr(dummy_node, key, value)

        results.append(dummy_node)

    return results
