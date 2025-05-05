import pytest
import networkx as nx

from src.graph.centralities.k_truss import node_k_truss_scores, cal_k_truss


def test_k_truss():
    # Same small graph for k-truss testing
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)])

    # In the triangle: edges (0, 1), (1, 2), (2, 0): one triangle, so 1 = (3-2) = (k-2). Then k=3 -> normalized = 3/3 = 1.0;
    # tail edges (1, 3), (3, 4). No trianles, 0 = (2-2) = (k-2). So, k=truss=2 -> normalized = 2/3

    expected = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 2/3,
        4: 2/3,
    }

    result = node_k_truss_scores(G)

    for node, exp in expected.items():
        assert result[node] == pytest.approx(exp, rel=1e-3)

    result = cal_k_truss(G)
    for node, exp in expected.items():
        assert result[node] == pytest.approx(exp, rel=1e-3)


def test_k_truss_triangle_tail():
    # a bit more complex graph for k-truss testing
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4),
                     (5, 0), (5, 1), (5, 2), (6, 3), (6, 4), (7, 6)])
    expected = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 3/4,
        4: 3/4,
        5: 1.0,
        6: 3/4,
        7: 0.5,
    }

    result = node_k_truss_scores(G)

    for node, exp in expected.items():
        assert result[node] == pytest.approx(exp, rel=1e-3)

    result = cal_k_truss(G)

    for node, exp in expected.items():
        assert result[node] == pytest.approx(exp, rel=1e-3)
