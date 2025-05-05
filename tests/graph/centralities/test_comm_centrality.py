import pytest
import networkx as nx

from src.graph.centralities.comm_centrality import comm_centrality
from src.graph.graph_utils import attach_communities


def test_comm_centrality():
    # Same small graph for comm_centrality testing
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])

    communities = [[0, 1, 2], [3, 4]]

    community_labels = attach_communities(G, communities)
    result = comm_centrality(G, community_labels)

    # node 1 should be the highest
    assert result[1] == pytest.approx(1.0, rel=1e-3)
