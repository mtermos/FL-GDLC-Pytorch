import numpy as np
from scipy.sparse import csr_matrix

# from https://github.com/tmagelinski/modularity_vitality/blob/master/src/analysis/modularity_vitality.py
# Thomas Magelinski


def hm_rescale(dict):
    """
    That little helper is simply doing a “divide-by-max” normalization:
    1. it takes your raw scores
    2. finds the largest one
    3. rescales everything so that the top score becomes 1 and the rest lie in (0,1]
    """
    max_factor = max(dict.values())
    x = {}
    for key, value in dict.items():
        x[key] = value / max_factor
    return x


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
