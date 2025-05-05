import pytest
import networkx as nx

from src.graph.centralities.k_core import cal_k_core


def test_k_core_triangle_tail():
    # Same small graph for k-core testing
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])
    # G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])
    # Our implementation normalizes core numbers
    result = cal_k_core(G)
    # Nodes [0,1,2] belong to the 2-core, while nodes [3,4] belong to the 1-core.
    # So normalized to 1.0 for [0,1,2], and 0.5 for [3,4].
    expected = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5}
    assert result == expected
