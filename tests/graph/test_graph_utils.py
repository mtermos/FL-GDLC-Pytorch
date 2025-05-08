import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
import pytest

from src.graph.graph_utils import (
    separate_graph,
    build_clean_graph,
    needs_community,
    detect_communities,
    attach_communities,
    separate_graphs_if_needed,
)


def frozenset_edges(G):
    return {frozenset(e) for e in G.edges()}


def test_separate_graph_simple():
    # Build a 4‐node cycle with a cross edge
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])
    communities = [{1, 2}, {3, 4}]

    intra, inter = separate_graph(G, communities)

    # All nodes should still be present
    assert set(intra.nodes()) == set(G.nodes())
    assert set(inter.nodes()) == set(G.nodes())

    # Intra‐community edges: only (1,2) and (3,4)
    assert frozenset_edges(intra) == {frozenset({1, 2}), frozenset({3, 4})}

    # Inter‐community edges: only (2,3) and (1,4)
    assert frozenset_edges(inter) == {frozenset({2, 3}), frozenset({1, 4})}


def test_build_clean_graph_and_labels(tmp_path):
    # Create a small edge list
    df = pd.DataFrame({
        "src": ["A", "B", "C"],
        "dst": ["B", "C", "A"]
    })
    G = build_clean_graph(df, "src", "dst", create_using=nx.Graph())

    # Should have exactly the nodes in the DF (no isolates)
    assert set(G.nodes()) == {"A", "B", "C"}
    # And the three edges forming a triangle
    assert frozenset_edges(G) == {
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
        frozenset({"C", "A"}),
    }

    # Each node must carry its own label attribute
    for n in G.nodes():
        assert G.nodes[n]["label"] == n


@pytest.mark.parametrize(
    "measures, expected",
    [
        ([], False),
        (["accuracy", "loss"], False),
        (["global_degree"], True),
        (["local_closeness", "foo"], True),
        (["Comm"], True),
        (["mv"], True),
    ],
)
def test_needs_community(measures, expected):
    assert needs_community(measures) is expected


def test_detect_communities_default_and_attach():
    # Two disconnected edges -> communities should be [{1,2}, {3,4}]
    G = nx.Graph()
    G.add_edges_from([(1, 2), (3, 4)])

    communities, G1, part = detect_communities(G)

    # Ensure we got two communities covering all nodes
    comm_sets = [set(c) for c in communities]
    assert set(map(frozenset, comm_sets)) == {
        frozenset({1, 2}),
        frozenset({3, 4})
    }

    # G1 should be an igraph Graph, part a Clustering
    assert isinstance(G1, ig.Graph)
    assert hasattr(part, "membership")

    # Now test attach_communities
    mapping = attach_communities(G, communities, attr_name="comm_idx")

    # Check that each node got its community index in both the mapping and node attrs
    for node, comm_idx in mapping.items():
        assert G.nodes[node]["comm_idx"] == comm_idx
    # The mapping returned should match what ended up on G
    assert mapping == {n: G.nodes[n]["comm_idx"] for n in G.nodes()}


def test_detect_communities_with_precomputed_part():
    # Build a trivial graph of two isolated nodes
    G = nx.Graph()
    G.add_nodes_from([10, 20])

    # Manually build an igraph with two vertices,
    # set labels to match the NetworkX nodes
    G1 = ig.Graph(n=2)
    G1.vs["label"] = [10, 20]

    # Create a VertexClustering that puts each node in its own community
    membership = [0, 1]
    part = ig.VertexClustering(G1, membership)

    communities, G1_out, part_out = detect_communities(G, G1=G1, part=part)

    # Should return what we passed in
    assert G1_out is G1
    assert part_out is part
    assert communities == [[10], [20]]


def test_separate_graphs_if_needed_empty():
    G = nx.Graph()
    G.add_edge("x", "y")
    intra, inter = separate_graphs_if_needed(G, communities=[])
    # When no communities given, both graphs should be the original
    assert intra is G
    assert inter is G


def test_separate_graphs_if_needed_nonempty():
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c")])
    communities = [{"a", "b"}, {"c"}]

    intra, inter = separate_graphs_if_needed(G, communities)
    # Only ("a","b") is intra, ("b","c") is inter
    assert frozenset_edges(intra) == {frozenset({"a", "b"})}
    assert frozenset_edges(inter) == {frozenset({"b", "c"})}
