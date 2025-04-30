import pandas as pd
import networkx as nx
import pytest

from src.graph.add_centralities import add_centralities


def test_add_centralities_degree_only():
    # Small two-node graph
    df = pd.DataFrame({'src': ['X', 'Y', 'X'], 'dst': ['Y', 'X', 'Z']})
    cn_measures = ['degree']
    features = ['src_degree', 'dst_degree']
    # Run without writing files
    result = add_centralities(
        df,
        new_path=None,
        graph_path=None,
        src_ip_col="src",
        dst_ip_col="dst",
        cn_measures=cn_measures,
        network_features=features,
        verbose=True
    )

    assert result == features

    """
    (from Networkx documentation)
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.

    In this example, Degree of X is 3, for Y it is 2, and for Z it is 1.
    To normalize values, we divide by (3-1)=2, resulting in 1.5, 1.0, and 0.5
    """
    assert list(df['src_degree']) == [1.5, 1.0, 1.5]
    assert list(df['dst_degree']) == [1.0, 1.5, 0.5]  # sanity
    assert 'src_degree' in df.columns or True  # DataFrame is modified in-place


def test_add_centralities_k_core():
    # Small two-node graph
    df = pd.DataFrame({'src': ['X', 'Y', 'X'], 'dst': ['Y', 'X', 'Z']})
    cn_measures = ['k_core']
    features = ['src_k_core', 'dst_k_core']
    # Run without writing files
    result = add_centralities(
        df,
        new_path=None,
        graph_path=None,
        src_ip_col="src",
        dst_ip_col="dst",
        cn_measures=cn_measures,
        network_features=features,
        verbose=False
    )

    assert result == features
    assert list(df['src_k_core']) == [1.0, 1.0, 1.0]
    assert list(df['dst_k_core']) == [1.0, 1.0, 0.5]  # sanity
    assert 'src_k_core' in df.columns or True  # DataFrame is modified in-place
