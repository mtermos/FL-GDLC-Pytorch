import random
from collections import defaultdict

from src.graph.centralities.centrality_utils import hm_rescale


def compute_degrees(G):
    """
    Return dict node -> degree(G, node).
    """
    return dict(G.degree())


def compute_in_out_degrees(G, partition):
    """
    Return two dicts:
      in_deg[node]  = # neighbors of node in the same community
      out_deg[node] = # neighbors of node in a different community
    """
    degrees = compute_degrees(G)
    in_deg = {}
    out_deg = {}

    for node, comm in partition.items():
        nbrs = set(G.neighbors(node))
        # Count how many neighbors share the same community
        same_comm = sum(1 for nbr in nbrs if partition.get(nbr) == comm)
        in_deg[node] = same_comm
        out_deg[node] = degrees[node] - same_comm

    return in_deg, out_deg


def compute_community_stats(partition, in_deg, out_deg, degrees):
    """
    Build community-level statistics:
      mu[comm]      = avg fraction of outer‐edges per node in comm
      max_in[comm]  = max in_deg among nodes in comm
      max_out[comm] = max out_deg among nodes in comm
    """
    comm2nodes = defaultdict(list)
    for node, comm in partition.items():
        comm2nodes[comm].append(node)

    mu = {}
    max_in = {}
    max_out = {}

    for comm, nodes in comm2nodes.items():
        # fraction of external edges per node
        fracs = [
            (out_deg[n] / degrees[n]) if degrees[n] > 0 else 0
            for n in nodes
        ]
        mu[comm] = (sum(fracs) / len(fracs)) if fracs else 0
        max_in[comm] = max(in_deg[n] for n in nodes) if nodes else 0
        max_out[comm] = max(out_deg[n] for n in nodes) if nodes else 0

    return mu, max_in, max_out


def comm_centrality(G, partition, R=None):
    """
    Community-aware centrality:
      C(i) = (1+μ_c) * (k_in(i)/max_in_c) * R
           + (1-μ_c) * [ (k_out(i)/max_out_c) * R ]^2

    Finally rescales all C(i) into [0,1] via hm_rescale.
    """
    degrees = compute_degrees(G)
    in_deg, out_deg = compute_in_out_degrees(G, partition)
    mu, max_in, max_out = compute_community_stats(
        partition, in_deg, out_deg, degrees
    )

    # single random scaling factor
    if not R:
        R = random.randint(1, 50)

    raw_scores = {}
    for node, comm in partition.items():
        kin, kout = in_deg[node], out_deg[node]
        mkin = max_in.get(comm, 1) or 1
        mkout = max_out.get(comm, 1) or 1
        alpha = mu.get(comm, 0)

        term_in = (kin / mkin) * R
        term_out = (kout / mkout) * R

        raw_scores[node] = (1 + alpha) * term_in \
            + (1 - alpha) * (term_out ** 2)

    # map into [0,1]
    return hm_rescale(raw_scores)
