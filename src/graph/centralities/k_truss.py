import igraph as ig
from src.graph.centralities.centrality_utils import hm_rescale

import networkx as nx


def node_k_truss_scores(G):
    """
    version 1
    """
    H = G.to_undirected().copy()
    truss = {}   # will hold the final trussness per edge
    k = 3

    while H.number_of_edges():
        T = nx.k_truss(H, k)
        dropped = set(H.edges()) - set(T.edges())
        # assign dropped edges to (k-1)-truss
        for e in dropped:
            truss[e] = k - 1
        H = T
        k += 1

    # now build node→max-edge-truss
    node_max = {n: 0 for n in G.nodes()}
    for (u, v), t in truss.items():
        node_max[u] = max(node_max[u], t)
        node_max[v] = max(node_max[v], t)

    return hm_rescale(node_max)


def cal_k_truss(G):
    """
    version 2: faster
    """
    sr_node_ktruss_dict = {}
    n = G.number_of_nodes()
    G = ig.Graph.from_networkx(G)
    ktrussdict = ktruss(G)
    nodetruss = [0] * n
    for edge in G.es:
        source = edge.source
        target = edge.target
        if not (source == target):
            t = ktrussdict[(source, target)]
        else:
            t = 0
        nodetruss[source] = max(nodetruss[source], t)
        nodetruss[target] = max(nodetruss[target], t)
    d = {}
    node_index = 0
    node_truss_value = 0
    for idx, t in enumerate(nodetruss):
        name = G.vs[idx]["_nx_name"]
        d[name] = t
    # while (node_index < len(nodetruss)):
    #     d[G.vs[node_index]["_nx_name"]] = nodetruss[node_truss_value]
    #     node_truss_value = node_truss_value+1
    #     node_index = node_index+1
    # print(d)
    return hm_rescale(d)


def edge_support(G):
    nbrs = dict((v.index, set(G.successors(v)) | set(G.predecessors(v)))
                for v in G.vs)
    support = {}
    for e in G.es:
        nod1, nod2 = e.source, e.target
        nod1_nbrs = set(nbrs[nod1])-set([nod1])
        nod2_nbrs = set(nbrs[nod2])-set([nod2])
        sup = len(nod1_nbrs.intersection(nod2_nbrs))
        support[(nod1, nod2)] = sup
    return support


def ktruss(G):
    support = edge_support(G)
    edges = sorted(support, key=support.get)  # type: ignore
    bin_boundaries = [0]
    curr_support = 0
    for i, e in enumerate(edges):
        if support[e] > curr_support:
            bin_boundaries.extend([i]*(support[e]-curr_support))
            curr_support = support[e]

    edge_pos = dict((e, pos) for pos, e in enumerate(edges))

    truss = {}
    neighbors = G.neighborhood()

    nbrs = dict(
        (v.index, (set(neighbors[v.index])-set([v.index]))) for v in G.vs)

    for e in edges:
        u, v = e[0], e[1]
        if not (u == v):
            common_nbrs = set(nbrs[u]).intersection(nbrs[v])
            for w in common_nbrs:
                if (u, w) in support:
                    e1 = (u, w)
                else:
                    e1 = (w, u)
                if (v, w) in support:
                    e2 = (v, w)
                else:
                    e2 = (w, v)
                pos = edge_pos[e1]
                if support[e1] > support[e]:
                    bin_start = bin_boundaries[support[e1]]
                    edge_pos[e1] = bin_start
                    edge_pos[edges[bin_start]] = pos
                    edges[bin_start], edges[pos] = edges[pos], edges[bin_start]
                    bin_boundaries[support[e1]] += 1

                pos = edge_pos[e2]
                if support[e2] > support[e]:
                    bin_start = bin_boundaries[support[e2]]
                    edge_pos[e2] = bin_start
                    edge_pos[edges[bin_start]] = pos
                    edges[bin_start], edges[pos] = edges[pos], edges[bin_start]
                    bin_boundaries[support[e2]] += 1

                support[e1] = max(support[e], support[e1]-1)
                support[e2] = max(support[e], support[e2]-1)

            truss[e] = support[e] + 2
            if v in nbrs[u]:
                nbrs[u].remove(v)
            if u in nbrs[v]:
                nbrs[v].remove(u)
    return truss
