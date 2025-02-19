import os
import sys
import itertools

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import networkx as nx


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src import *
from src.utils.define_graph import define_graph

colors = [
    (0, "white"),
    (0.5, "blue"),
    (1, "red"),
    # (-1, "red"),
]  # Replace 'blue' and 'red' with your desired colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


def reorder_adj(A, edge_index):
    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G)
    comminity_label = np.zeros(A.shape[0], dtype=np.int32)
    for id, c in enumerate(community):
        for node in c:
            comminity_label[node] = id
    # idx = np.argsort(comminity_label)

    sorted_community_groups = sorted(
        community, key=lambda item: len(item), reverse=True
    )
    community_based_node_order = list(
        itertools.chain.from_iterable(sorted_community_groups)
    )

    # dense_abar = dense_abar[:, idx]
    # dense_abar = dense_abar[idx, :]
    A = A[:, community_based_node_order]
    A = A[community_based_node_order, :]

    return A


def estimate_p():
    graph = define_graph(config.dataset.dataset_name)
    n = graph.num_nodes
    edge_index = graph.edge_index

    dbar = 2 * edge_index.shape[1] / (n - 1)
    p = dbar / n

    return p, n
