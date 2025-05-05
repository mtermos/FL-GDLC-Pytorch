import time
import logging
import networkx as nx
import igraph as ig

from src.graph.graph_utils import build_clean_graph, needs_community, detect_communities, attach_communities, separate_graphs_if_needed
from src.graph.centralities.betweenness import cal_betweenness_centrality
from src.graph.centralities.k_core import cal_k_core
from src.graph.centralities.k_truss import cal_k_truss
from src.graph.centralities.comm_centrality import comm_centrality
from src.graph.centralities.modularity_vitality import modularity_vitality


logger = logging.getLogger(__name__)


class CentralitySpec:
    def __init__(self, func, graph, require_simple_graph=False, extra={}):
        self.func = func
        self.graph = graph
        self.require_simple_graph = require_simple_graph
        self.extra = extra


def timed_set_node_attribute(G, name, func, graph=None, verbose=False, **func_kwargs):
    """
    Compute centrality with `func`, time it, set as node attribute `name`, 
    and return the resulting dict.

    Parameters
    ----------
    G          : original NetworkX graph (for setting node attributes)
    name       : attribute name to set on the nodes of G 
    func       : callable that returns {node: value}
    graph      : the graph to pass to func (defaults to G)
    verbose    : if True, logs start, end, and elapsed time
    func_kwargs: extra kwargs to pass to func
    """
    graph = graph or G
    if verbose:
        logger.info(f"Starting {name!r} centrality...")
    t0 = time.perf_counter()
    centrality = func(graph, **func_kwargs)
    elapsed = time.perf_counter() - t0
    if verbose:
        logger.info(f"Finished {name!r} in {elapsed:.2f}s")
    nx.set_node_attributes(G, centrality, name)
    return centrality


def add_centralities(df, new_path, graph_path, src_ip_col, dst_ip_col, cn_measures,
                     network_features, G=None, create_using=nx.DiGraph(),
                     communities=None, G1=None, part=None, verbose=False):

    if not network_features or not cn_measures:
        return []
    # 1) Build or reuse G
    if G is None:
        G = build_clean_graph(df, src_ip_col, dst_ip_col, create_using)

    # 2) Community detection if needed
    if needs_community(cn_measures) and communities is None:
        communities, G1, part = detect_communities(G, G1, part)

    if communities:
        community_labels = attach_communities(G, communities)

    intra_graph, inter_graph = separate_graphs_if_needed(G, communities)

    # 3) Determine if the graph is “simple”
    multi_graph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))

    # 4) A mapping from measure → (function, which subgraph to use, extra kwargs)
    CENTRALITY_SPECS = {
        "betweenness":      CentralitySpec(cal_betweenness_centrality, G),
        "local_betweenness": CentralitySpec(cal_betweenness_centrality, intra_graph),
        "global_betweenness": CentralitySpec(cal_betweenness_centrality, inter_graph),
        "degree":           CentralitySpec(nx.degree_centrality, G),
        "local_degree":     CentralitySpec(nx.degree_centrality, intra_graph),
        "global_degree":    CentralitySpec(nx.degree_centrality, inter_graph),
        "eigenvector":      CentralitySpec(nx.eigenvector_centrality, G, True, {"max_iter": 600}),
        "local_eigenvector": CentralitySpec(nx.eigenvector_centrality, intra_graph, True, {"max_iter": 600}),
        "global_eigenvector": CentralitySpec(nx.eigenvector_centrality, inter_graph, True, {"max_iter": 600}),
        "closeness":        CentralitySpec(nx.closeness_centrality, G),
        "local_closeness":  CentralitySpec(nx.closeness_centrality, intra_graph),
        "global_closeness": CentralitySpec(nx.closeness_centrality, inter_graph),
        "pagerank":         CentralitySpec(nx.pagerank, G, False, {"alpha": 0.85}),
        "local_pagerank":   CentralitySpec(nx.pagerank, intra_graph, False, {"alpha": 0.85}),
        "global_pagerank":  CentralitySpec(nx.pagerank, inter_graph, False, {"alpha": 0.85}),
        "k_core":           CentralitySpec(cal_k_core, G, True),
        "k_truss":          CentralitySpec(cal_k_truss, G),
        "Comm":             CentralitySpec(lambda g: comm_centrality(g, community_labels), G),
        "mv":               CentralitySpec(lambda g: modularity_vitality(G1, part), G),
    }

    features_dicts = {}
    for measure in cn_measures:
        spec = CENTRALITY_SPECS.get(measure)
        if not spec:
            continue
        if multi_graph and spec.require_simple_graph:
            # skip if it must be simple but our graph isn’t
            continue
        features_dicts[measure] = timed_set_node_attribute(
            G,
            name=measure,
            func=spec.func,
            graph=spec.graph,
            verbose=verbose,
            **spec.extra
        )

    # 5) Persist graph
    if graph_path:
        nx.write_gexf(G, graph_path)

    # 6) Merge back into df
    for feature in network_features:
        prefix, attr = feature.split("_", 1)
        col = src_ip_col if prefix == "src" else dst_ip_col
        df[feature] = df[col].map(features_dicts.get(attr, {})).fillna(-1)

    # 7) Persist DataFrame
    if new_path:
        df.to_parquet(new_path)
        if verbose:
            logger.info(f"DataFrame written to {new_path}")

    return network_features
