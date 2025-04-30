gdlc_centrality_measures = [
    ["betweenness", "local_betweenness", "degree", "local_degree",
     "eigenvector", "closeness", "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "degree", "global_degree",
     "eigenvector", "closeness", "pagerank", "global_pagerank", "k_core", "k_truss", "mv"],
    ["betweenness", "local_betweenness", "pagerank",
        "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "pagerank",
        "global_pagerank", "k_core", "k_truss", "mv"]
]

gdlc_network_features = [
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_degree', 'dst_degree', 'src_global_degree', 'dst_global_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],
]
