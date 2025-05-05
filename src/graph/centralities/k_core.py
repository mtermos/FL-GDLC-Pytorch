import networkx as nx
from src.graph.centralities.centrality_utils import hm_rescale


def cal_k_core(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore_dict_eachNode = nx.core_number(G)
    kcore_dict_eachNode_normalized = hm_rescale(kcore_dict_eachNode)
    return kcore_dict_eachNode_normalized
