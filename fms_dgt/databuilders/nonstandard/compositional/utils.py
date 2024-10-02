# Standard
from typing import Dict, List


def extract_exec_sets(graph: Dict) -> List:
    leaf_nodes = [val for v in graph.values() for val in v if val not in graph]
    # TODO: fix
    return [leaf_nodes, list(graph.keys())]
