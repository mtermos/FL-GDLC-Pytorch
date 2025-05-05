import pytest
import networkx as nx
import igraph as ig

from src.graph.centralities.modularity_vitality import modularity_vitality
from src.graph.graph_utils import build_clean_graph, needs_community, detect_communities, attach_communities, separate_graphs_if_needed


def test_modularity_vitality():
    # Same small graph for modularity_vitality testing
    G = nx.DiGraph()
    G.add_edges_from([("a", 1), (1, 2), (2, "a"), (1, 3), (3, 4)])

    # communities, G1, part = detect_communities(G, G1=None, part=None)
    # print(f"==>> communities: {communities}")
    communities = [["a", 1, 2], [3, 4]]
    G1 = ig.Graph.from_networkx(G)
    name_to_idx = {v["_nx_name"]: v.index for v in G1.vs}

    # prepare the membership list
    membership = [None] * G1.vcount()

    # fill it by translating each name to its index
    for cid, comm in enumerate(communities):
        for vid in comm:
            idx = name_to_idx.get(vid)
            if idx is None:
                raise KeyError(f"Node {vid!r} not found in graph")
            membership[idx] = cid

    # optional: check you covered every vertex
    if any(m is None for m in membership):
        missing = [i for i, m in enumerate(membership) if m is None]
        raise ValueError(f"Some vertices have no community: {missing}")

    # now build your VertexClustering
    part = ig.VertexClustering(G1, membership)

    result = modularity_vitality(G1, part)

    """
    In the original implementation node 4 should be the highest; removing it would make the rest one community.
    and node 1 has a negative value.

    I changed the sorting so node 1 will have the highest value.
    It affects the modularity negatively, then it is important in the structure of the graph
    """
    assert result[1] == pytest.approx(1.0, rel=1e-3)
