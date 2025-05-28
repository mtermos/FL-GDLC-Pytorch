import networkx as nx
import igraph as ig


def separate_graph(graph, communities):
    """
    Separates a graph into intra-community and inter-community edges.

    Parameters:
    - graph: A NetworkX graph
    - communities: A list of sets, where each set contains the nodes in a community

    Returns:
    - intra_graph: A graph containing only intra-community edges
    - inter_graph: A graph containing only inter-community edges
    """
    # Create new graphs for intra-community and inter-community edges
    intra_graph = nx.Graph()
    inter_graph = nx.Graph()

    # Add all nodes to both graphs to ensure structure is maintained
    intra_graph.add_nodes_from(graph.nodes())
    inter_graph.add_nodes_from(graph.nodes())

    # Organize communities in a way that allows quick lookup of node to community mapping
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Iterate through each edge in the original graph
    for edge in graph.edges():
        node_u, node_v = edge

        # Determine if an edge is intra-community or inter-community
        if node_to_community.get(node_u) == node_to_community.get(node_v):
            # Intra-community edge
            intra_graph.add_edge(node_u, node_v)
        else:
            # Inter-community edge
            inter_graph.add_edge(node_u, node_v)

    return intra_graph, inter_graph


def build_clean_graph(df, src_ip_col, dst_ip_col, create_using):
    """
    Construct a NetworkX graph from a pandas DataFrame, remove isolates,
    and set a 'label' attribute on each node.
    """
    G = nx.from_pandas_edgelist(
        df,
        source=src_ip_col,
        target=dst_ip_col,
        create_using=create_using
    )
    # Drop isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    # Retain original node identity as a 'label'
    for node in G.nodes():
        G.nodes[node]['label'] = node
    return G


def needs_community(cn_measures):
    """
    Return True if any requested centrality measure requires community detection.
    """
    comm_list = [
        "local_betweenness", "global_betweenness",
        "local_degree",     "global_degree",
        "local_eigenvector", "global_eigenvector",
        "local_closeness",  "global_closeness",
        "local_pagerank",   "global_pagerank",
        "Comm", "mv"
    ]
    return any(m in comm_list for m in cn_measures)


def detect_communities(G, G1=None, part=None):
    """
    Perform Infomap community detection on G, returning:
      - communities (list of lists of node labels),
      - the igraph Graph G1,
      - the membership partition object `part`.
    """
    # part = nx.community.louvain_communities(G, seed=123)
    # communities = [list(comm) for comm in part]
    # return communities, None, None
    # Convert to igraph if not already done
    if G1 is None:
        G1 = ig.Graph.from_networkx(G)
        labels = [G.nodes[n].get('label', n) for n in G.nodes()]
        G1.vs['label'] = labels
    # Run Infomap if needed
    if part is None:
        part = G1.community_infomap()
    # Extract communities as lists of original node labels
    communities = [[G1.vs[idx]['label'] for idx in comm] for comm in part]
    return communities, G1, part


def attach_communities(G, communities, attr_name="new_community"):
    """
    Set a node attribute on G mapping each node to its community index.
    """
    community_labels = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_labels[node] = i
    nx.set_node_attributes(G, community_labels, attr_name)

    return community_labels


def separate_graphs_if_needed(G, communities):
    """
    Split G into intra-community and inter-community subgraphs
    if communities provided; otherwise return G for both.
    """
    if communities:
        intra_graph, inter_graph = separate_graph(G, communities)
    else:
        intra_graph = G
        inter_graph = G
    return intra_graph, inter_graph
