import yaml
import traceback
from pathlib import Path
from gqlalchemy import Memgraph

QUERIES_PATH = Path(__file__).parent / "queries.yaml"


class QueryRunner:
    def __init__(self, host: str, port: str):
        # First set up the queries map
        self.queries_path = QUERIES_PATH
        self.queries_map = yaml.safe_load(self.queries_path.read_text())

        # Set up memgrpah instance
        print(host)
        print(port)
        self.memgraph = Memgraph(host=host, port=port)

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
        results = self.memgraph.execute_and_fetch(query)
        
        return [result[n_or_m] for result in results]
