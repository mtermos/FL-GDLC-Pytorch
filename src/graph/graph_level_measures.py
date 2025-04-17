import os
import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig
import json


def compute_graph_properties(G, name=None, folder_path=None):

    G.remove_nodes_from(list(nx.isolates(G)))

    for node in G.nodes():
        G.nodes[node]['label'] = node

    G1 = ig.Graph.from_networkx(G)

    labels = [G.nodes[node]['label'] for node in G.nodes()]
    G1.vs['label'] = labels

    part = G1.community_infomap()

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['label'] for node_index in com])

    properties = {}
    properties["name"] = name
    properties["number_of_nodes"] = G.number_of_nodes()
    properties["number_of_edges"] = G.number_of_edges()

    degrees = [degree for _, degree in G.degree()]
    properties["max_degree"] = max(degrees)
    properties["avg_degree"] = sum(degrees) / len(degrees)
    properties["transitivity"] = nx.transitivity(G)
    properties["density"] = nx.density(G)

    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    inter_cluster_edges = 0
    for u, v in G.edges():
        if node_to_community[u] != node_to_community[v]:
            inter_cluster_edges += 1

    properties["mixing_parameter"] = inter_cluster_edges / G.number_of_edges()

    if folder_path:
        filename = os.path.join(folder_path, f"graph_{name}.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as outfile:
            json.dump(properties, outfile)

    return properties
