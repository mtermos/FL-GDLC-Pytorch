import pytest
import networkx as nx

from src.graph.centralities.betweenness import cal_betweenness_centrality


def test_betweenness_triangle_tail():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])
    # Compute betweenness using our function
    result = cal_betweenness_centrality(G)
    # Compute expected using networkx
    expected = nx.betweenness_centrality(G, normalized=True)
    for node, value in expected.items():
        assert result[node] == pytest.approx(value, rel=1e-3)
