import igraph as ig
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import random

from sklearn.preprocessing import StandardScaler

from src.graph.graph_utils import betweenness_rescale, hm_rescale, separate_graph


def cal_betweenness_centrality(G):
    G1 = ig.Graph.from_networkx(G)
    estimate = G1.betweenness(directed=True)
    b = dict(zip(G1.vs["_nx_name"], estimate))
    return betweenness_rescale(b, G1.vcount(), True)


def cal_k_core(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore_dict_eachNode = nx.core_number(G)

    kcore_dict_eachNode_normalized = hm_rescale(kcore_dict_eachNode)

    return kcore_dict_eachNode_normalized


def cal_k_truss(G):
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
    while (node_index < len(nodetruss)):
        d[G.vs[node_index]["_nx_name"]] = nodetruss[node_truss_value]
        node_truss_value = node_truss_value+1
        node_index = node_index+1
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

##########################################
##########################################
##########################################
##########################################


def degree_method(g):
    list_nodes = g.nodes()  # list of nodes
    dict_degree = dict()  # nodes and their degree centrality

    for i in list_nodes:
        dict_degree[i] = g.degree(i)

    return dict_degree


def fraction_outerconnections(g, best):
    dict_fraction_outerconnections = dict()
    list = g.nodes()

    # best = c.best_partition (g)
    list_community = []
    for i in best:
        list_community.append(best[i])
    list_community.sort()
    list_community = set(list_community)
    l = []
    for i in list_community:
        l.append(i)
    list_community = l  # all communities in a list [0, 1, .... x]

    degree = degree_method(g)
    degree_in, degree_out = inout_degree(g, best)

    for i in list_community:  # for each community
        µ = 0
        couter = 0
        for j in list:  # for each node
            if j not in best:
                continue
            if best[j] == i:  # if the community of node j is in community i
                couter = couter+1  # increment counter
                kout = degree_out[j]  # get the outer degrees of this node
                k = degree[j]  # get the total degrees of this node
                µ = µ+float(kout/k)  # add it to u
        if couter > 0:
            dict_fraction_outerconnections[i] = float(µ/couter)
        else:
            dict_fraction_outerconnections[i] = 0
    return dict_fraction_outerconnections


def inout_degree(g, best):
    list = g.nodes()
    dict_graph = dict()  # nodes in the key and their neighbors
    for i in list:
        dict_graph[i] = []
    for i in list:
        iteri = g.neighbors(i)
        for j in iteri:
            dict_graph[i].append(j)

    dict_in = dict()
    dict_out = dict()
    d = degree_method(g)

    for i in list:
        if i not in best:
            continue
        community = best[i]
        nbr_in = 0
        for j in dict_graph[i]:
            if best[j] == community:
                # intra links, since node's community (best[i]) is in the same as j's (best[j])
                nbr_in = nbr_in + 1
        idegree = d[i]
        nbr_out = idegree - nbr_in  # inter links
        dict_in[i] = nbr_in
        dict_out[i] = nbr_out

    return dict_in, dict_out


def max_inout(g, partition):
    # partition: dict(node: its community)

    max_in = dict()
    max_out = dict()
    list = g.nodes()

    list_community = []
    for i in partition:
        list_community.append(partition[i])
    list_community.sort()
    list_community = set(list_community)
    l = []
    for i in list_community:
        l.append(i)
    list_community = l

    degree_in, degree_out = inout_degree(g, partition)

    for i in list_community:
        inmax = outmax = 0
        for j in list:
            if j not in partition:
                continue
            if partition[j] == i:
                if inmax < degree_in[j]:
                    inmax = degree_in[j]
                if outmax < degree_out[j]:
                    outmax = degree_out[j]
        max_in[i] = inmax
        max_out[i] = outmax

    return max_in, max_out


def min_max_scaling(centrality_scores: dict):
    """
    Perform Min-Max scaling on a list of centrality scores.

    Parameters:
    centrality_scores (list): List of centrality scores to be scaled.

    Returns:
    list: Scaled centrality scores between 0 and 1.
    """
    if len(centrality_scores) == 0:
        return {}
    min_score = min(centrality_scores.values())
    max_score = max(centrality_scores.values())

    scaled_scores = {}
    for i in centrality_scores:
        scaled_scores[i] = (centrality_scores[i] -
                            min_score) / (max_score - min_score)

    return scaled_scores


def comm_centrality(g, partition):

    list = g.nodes()
    dict_comm = dict()

    max_in, max_out = max_inout(g, partition)
    dict_fraction_outerconnections = fraction_outerconnections(g, partition)
    dict_in, dict_out = inout_degree(g, partition)
    R = random.randint(1, 50)
    for i in list:
        if i not in partition:
            continue
        community = partition[i]
        µc = dict_fraction_outerconnections[community]
        maxkin = max_in[community]
        maxkout = max_out[community]
        kin = dict_in[i]
        kout = dict_out[i]

        if maxkout == 0:
            dict_comm[i] = (1 + µc) * (float(kin / maxkin) * R) + (1 - µc) * (float(kout) * R) * (
                float(kout) * R)
        else:
            try:
                dict_comm[i] = (1 + µc) * (float(kin / maxkin) * R) + (1 - µc) * (float(kout / maxkout) * R) * (
                    float(kout / maxkout) * R)
            except:
                pass

    d = dict_comm
    l = []
    for i in d:
        l.append(i)

    n = len(d)
    for i in range(0, n - 1):
        max = i
        for j in range(i + 1, n):
            if d[l[max]] < d[l[j]]:
                max = j
        if max != i:
            x = l[i]
            l[i] = l[max]
            l[max] = x

    dict_tri = dict()
    for i in l:
        dict_tri[i] = d[i]

    return min_max_scaling(dict_tri)


##########################################
##########################################
##########################################
##########################################


def getSparseA(g):
    edges = [(e.source, e.target) for e in g.es()]
    sources, targets = list(zip(*edges))
    if g.is_weighted():
        weights = np.array(g.es['weight'], dtype=float)
    else:
        weights = np.ones(len(sources))
    self_loop_inds = (np.array(sources) == np.array(targets))
    weights[self_loop_inds] = weights[self_loop_inds] / 2
    weights = list(weights)
    A = csr_matrix((weights + weights, (sources + targets, targets + sources)),
                   shape=(g.vcount(), g.vcount()))
    return A


def getGroupIndicator(g, membership, rows=None):
    if not rows:
        rows = list(range(g.vcount()))
    cols = membership
    vals = np.ones(len(cols))
    group_indicator_mat = csr_matrix((vals, (rows, cols)),
                                     shape=(g.vcount(), max(membership) + 1))
    return group_indicator_mat


def getDegMat(node_deg_by_group, rows, cols):
    degrees = node_deg_by_group.sum(1)
    degrees = np.array(degrees).flatten()
    deg_mat = csr_matrix((degrees, (rows, cols)),
                         shape=node_deg_by_group.shape)
    degrees = degrees[:, np.newaxis]
    return degrees, deg_mat


def newMods(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None
    index = list(range(g.vcount()))
    membership = part.membership

    m = sum(g.strength(weights=weight_key)) / 2

    A = getSparseA(g)
    self_loops = A.diagonal().sum()
    group_indicator_mat = getGroupIndicator(g, membership, rows=index)
    node_deg_by_group = A * group_indicator_mat

    internal_edges = (
        node_deg_by_group[index, membership].sum() + self_loops) / 2

    degrees, deg_mat = getDegMat(node_deg_by_group, index, membership)
    node_deg_by_group += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = node_deg_by_group[index, membership].transpose() - degrees

    starCenter = (degrees == m)
    degrees[starCenter] = 0  # temp replacement avoid division by 0

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    # expanding out (group_degs - node_deg_by_group)^2 is slightly faster:
    expected_impact = np.power(group_degs, 2).sum() - 2 * (node_deg_by_group * group_degs.transpose()) +\
        node_deg_by_group.multiply(node_deg_by_group).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)
    q1s = q1_links - q1_degrees
    q1s[starCenter] = 0
    q1s = np.array(q1s).flatten()
    return q1s


def modularity_vitality(g, part):
    q0 = part.modularity
    q1s = newMods(g, part)
    vitalities = (q0 - q1s).tolist()
    m = {}
    for i in range(len(vitalities)):
        m[g.vs[i]['label']] = vitalities[i]
    return m


##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################


def add_centralities(df, new_path, graph_path, dataset, cn_measures, network_features, G=None, create_using=nx.DiGraph(), communities=None, G1=None, part=None):

    if not G:
        print("constructing graph")
        G = nx.from_pandas_edgelist(
            df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=create_using)
        G.remove_nodes_from(list(nx.isolates(G)))
        for node in G.nodes():
            G.nodes[node]['label'] = node

    need_communities = False
    comm_list = ["local_betweenness", "global_betweenness", "local_degree", "global_degree", "local_eigenvector",
                 "global_eigenvector", "local_closeness", "global_closeness", "local_pagerank", "global_pagerank", "Comm", "mv"]
    if any(value in comm_list for value in cn_measures):
        need_communities = True

    if need_communities and not communities:
        print("calculating communities")
        if not G1:
            G1 = ig.Graph.from_networkx(G)
            labels = [G.nodes[node].get('label', 'default_label')
                      for node in G.nodes()]
            G1.vs['label'] = labels
        if not part:
            part = G1.community_infomap()

        communities = []
        for com in part:
            communities.append([G1.vs[node_index]['label']
                               for node_index in com])
    if communities:
        community_labels = {}
        for i, community in enumerate(communities):
            for node in community:
                community_labels[node] = i

        nx.set_node_attributes(G, community_labels, "new_community")

        intra_graph, inter_graph = separate_graph(G, communities)

    simple_graph = create_using != nx.MultiDiGraph
    if G:
        simple_graph = type(G) is not nx.MultiDiGraph

    if "betweenness" in cn_measures:
        print("calculating betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(G), "betweenness")
        print("calculated betweenness")
    if "local_betweenness" in cn_measures:
        print("calculating local_betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(
            intra_graph), "local_betweenness")
        print("calculated local_betweenness")
    if "global_betweenness" in cn_measures:
        print("calculating global_betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(
            inter_graph), "global_betweenness")
        print("calculated global_betweenness")
    if "degree" in cn_measures:
        print("calculating degree")
        nx.set_node_attributes(G, nx.degree_centrality(G), "degree")
        print("calculated degree")
    if "local_degree" in cn_measures:
        print("calculating local_degree")
        nx.set_node_attributes(
            G, nx.degree_centrality(intra_graph), "local_degree")
        print("calculated local_degree")
    if "global_degree" in cn_measures:
        print("calculating global_degree")
        nx.set_node_attributes(G, nx.degree_centrality(
            inter_graph), "global_degree")
        print("calculated global_degree")
    if "eigenvector" in cn_measures and simple_graph:
        print("calculating eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            G, max_iter=600), "eigenvector")
        print("calculated eigenvector")
    if "local_eigenvector" in cn_measures and simple_graph:
        print("calculating local_eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            intra_graph), "local_eigenvector")
        print("calculated local_eigenvector")
    if "global_eigenvector" in cn_measures and simple_graph:
        print("calculating global_eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            inter_graph), "global_eigenvector")
        print("calculated global_eigenvector")
    if "closeness" in cn_measures:
        print("calculating closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(G), "closeness")
        print("calculated closeness")
    if "local_closeness" in cn_measures:
        print("calculating local_closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(
            intra_graph), "local_closeness")
        print("calculated local_closeness")
    if "global_closeness" in cn_measures:
        print("calculating global_closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(
            inter_graph), "global_closeness")
        print("calculated global_closeness")
    if "pagerank" in cn_measures:
        print("calculating pagerank")
        nx.set_node_attributes(G, nx.pagerank(G, alpha=0.85), "pagerank")
        print("calculated pagerank")
    if "local_pagerank" in cn_measures:
        print("calculating local_pagerank")
        nx.set_node_attributes(G, nx.pagerank(
            intra_graph, alpha=0.85), "local_pagerank")
        print("calculated local_pagerank")
    if "global_pagerank" in cn_measures:
        print("calculating global_pagerank")
        nx.set_node_attributes(G, nx.pagerank(
            inter_graph, alpha=0.85), "global_pagerank")
        print("calculated global_pagerank")
    if "k_core" in cn_measures and simple_graph:
        print("calculating k_core")
        nx.set_node_attributes(G, cal_k_core(G), "k_core")
        print("calculated k_core")
    if "k_truss" in cn_measures:
        print("calculating k_truss")
        nx.set_node_attributes(G, cal_k_truss(G), "k_truss")
        print("calculated k_truss")
    if "Comm" in cn_measures:
        print("calculating Comm")
        nx.set_node_attributes(
            G, comm_centrality(G, community_labels), "Comm")
        print("calculated Comm")
    if "mv" in cn_measures:
        print("calculating mv")
        nx.set_node_attributes(G, modularity_vitality(G1, part), "mv")
        print("calculated mv")

    if graph_path:
        nx.write_gexf(G, graph_path)

    features_dicts = {}
    for measure in cn_measures:
        features_dicts[measure] = nx.get_node_attributes(G, measure)
        print(f"==>> features_dicts: {measure , len(features_dicts[measure])}")

    for feature in network_features:
        if feature[:3] == "src":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.src_ip_col], -1), axis=1)
        if feature[:3] == "dst":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.dst_ip_col], -1), axis=1)

    if new_path:
        df.to_parquet(new_path)
        print(f"DataFrame written to {new_path}")

    return network_features


def normalize_centrality(centrality_dict):
    # Extract values and reshape for sklearn
    values = np.array(list(centrality_dict.values())).reshape(-1, 1)

    # Apply z-score normalization
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(values).flatten()

    # Create a dictionary of normalized values
    normalized_centrality = {node: norm_value for node, norm_value in zip(
        centrality_dict.keys(), normalized_values)}

    return normalized_centrality


def add_centralities_as_node_features(df, G, graph_path, dataset, cn_measures, create_using=nx.DiGraph()):

    if not G:
        G = nx.from_pandas_edgelist(
            df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=create_using)

    G.remove_nodes_from(list(nx.isolates(G)))
    for node in G.nodes():
        G.nodes[node]['label'] = node

    compute_communities = False
    comm_list = ["local_betweenness", "global_betweenness", "local_degree", "global_degree", "local_eigenvector",
                 "global_eigenvector", "local_closeness", "global_closeness", "local_pagerank", "global_pagerank", "Comm", "mv"]
    if any(value in comm_list for value in cn_measures):
        compute_communities = True

    if compute_communities:
        G1 = ig.Graph.from_networkx(G)
        labels = [G.nodes[node]['label'] for node in G.nodes()]
        G1.vs['label'] = labels

        part = G1.community_infomap()
        communities = []
        for com in part:
            communities.append([G1.vs[node_index]['label']
                               for node_index in com])

        community_labels = {}
        for i, community in enumerate(communities):
            for node in community:
                community_labels[node] = i

        nx.set_node_attributes(G, community_labels, "new_community")

        intra_graph, inter_graph = separate_graph(G, communities)

    if "betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "betweenness")
        print("calculated betweenness")
    if "local_betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_betweenness")
        print("calculated local_betweenness")
    if "global_betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_betweenness")
        print("calculated global_betweenness")
    if "degree" in cn_measures:
        normalized_centrality = normalize_centrality(nx.degree_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "degree")
        print("calculated degree")
    if "local_degree" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.degree_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_degree")
        print("calculated local_degree")
    if "global_degree" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.degree_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_degree")
        print("calculated global_degree")
    if "eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(G, max_iter=600))
        nx.set_node_attributes(G, normalized_centrality, "eigenvector")
        print("calculated eigenvector")
    if "local_eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_eigenvector")
        print("calculated local_eigenvector")
    if "global_eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_eigenvector")
        print("calculated global_eigenvector")
    if "closeness" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.closeness_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "closeness")
        print("calculated closeness")
    if "local_closeness" in cn_measures:
        normalized_centrality = normalize_centrality(nx.closeness_centrality(
            intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_closeness")
        print("calculated local_closeness")
    if "global_closeness" in cn_measures:
        normalized_centrality = normalize_centrality(nx.closeness_centrality(
            inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_closeness")
        print("calculated global_closeness")
    if "pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.pagerank(G, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "pagerank")
        print("calculated pagerank")
    if "local_pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(nx.pagerank(
            intra_graph, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "local_pagerank")
        print("calculated local_pagerank")
    if "global_pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(nx.pagerank(
            inter_graph, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "global_pagerank")
        print("calculated global_pagerank")
    if "k_core" in cn_measures:
        normalized_centrality = normalize_centrality(cal_k_core(G))
        nx.set_node_attributes(G, normalized_centrality, "k_core")
        print("calculated k_core")
    if "k_truss" in cn_measures:
        normalized_centrality = normalize_centrality(cal_k_truss(G))
        nx.set_node_attributes(G, normalized_centrality, "k_truss")
        print("calculated k_truss")

    if "Comm" in cn_measures:
        print("calculating Comm")
        nx.set_node_attributes(
            G, comm_centrality(G, community_labels), "Comm")
        print("calculated Comm")
    if "mv" in cn_measures:
        print("calculating mv")
        nx.set_node_attributes(G, modularity_vitality(G1, part), "mv")
        print("calculated mv")

    if graph_path:
        nx.write_gexf(G, graph_path)

    return G
