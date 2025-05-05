import numpy as np
from scipy.sparse import diags

from src.graph.centralities.centrality_utils import hm_rescale, getSparseA, getGroupIndicator, getDegMat

# from https://github.com/tmagelinski/modularity_vitality/blob/master/src/analysis/modularity_vitality.py
# Thomas Magelinski


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


def invert_min_max_scale(raw_scores: dict) -> dict:
    """
    Linearly map raw_scores so that the most negative (min) → 1.0
    and the most positive (max) → 0.0, everything else in between.
    """
    if not raw_scores:
        return {}

    vmin = min(raw_scores.values())
    vmax = max(raw_scores.values())
    span = vmax - vmin
    if span == 0:
        # all scores identical → everyone gets 1.0 (or 0.0, your choice)
        return {node: 1.0 for node in raw_scores}

    return {
        node: (vmax - v) / span
        for node, v in raw_scores.items()
    }


def modularity_vitality(g, part):
    q0 = part.modularity
    q1s = newMods(g, part)
    vitalities = (q0 - q1s).tolist()
    try:
        names = g.vs["_nx_name"]
    except KeyError:
        names = g.vs["name"]
    vitalities = dict(zip(names, vitalities))
    return invert_min_max_scale(vitalities)
