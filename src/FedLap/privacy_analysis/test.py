import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import torch


from src import *
from src.utils.define_graph import define_graph
from src.FedLap.utils import *


def estimate_p_s_p_e(graph, community_groups=None):
    if community_groups is None:
        all_nodes = torch.arange(graph.num_nodes)
        community_groups = [
            all_nodes[graph.y == label] for label in range(graph.num_classes)
        ]

    internal_nodes = 0
    p_s = []
    p_e = []
    nis = []

    for subgraph_nodes in community_groups:
        if len(subgraph_nodes) < 10:
            continue
        node_ids = torch.tensor(subgraph_nodes)
        n_i = len(node_ids)

        # external_nodes = all_nodes[~all_nodes.unsqueeze(1).eq(node_ids).any(1)]
        edge_mask = graph.edge_index.unsqueeze(2).eq(node_ids).any(2).any(0)
        intra_mask = graph.edge_index.unsqueeze(2).eq(node_ids).any(2).all(0)
        all_edges = graph.edge_index[:, edge_mask]
        intra_edges = graph.edge_index[:, intra_mask]

        internal_edges_i = intra_edges.shape[1]
        external_edges = all_edges.shape[1] - internal_edges_i
        internal_nodes += internal_edges_i

        p_s.append(internal_edges_i / (2 * n_i * (n_i - 1)))
        p_e.append(external_edges / (2 * n_i * (graph.num_nodes - n_i)))
        nis.append(n_i)

    p_s_total = sum([p * n_i for p, n_i in zip(p_s, nis)]) / graph.num_nodes
    p_e_total = sum([p * n_i for p, n_i in zip(p_e, nis)]) / graph.num_nodes

    return p_s_total, p_e_total


if __name__ == "__main__":
    # graph = define_graph(config.dataset.dataset_name)
    # n = graph.num_nodes

    p, n = estimate_p()
    print(p, n)

    # edge_index = graph.edge_index
    # community_groups = find_community(graph.edge_index, graph.num_nodes)
    # G = nx.Graph(edge_index.T.tolist())
    # community = nx.community.louvain_communities(G)
    # community_groups = [list(com) for com in community]

    # all_nodes = torch.arange(n)
    # community_groups = [all_nodes[graph.y == label] for label in range(graph.num_classes)]

    # A = create_adj(edge_index, graph.num_nodes).to_dense()

    # internal_nodes = 0
    # p_s = []
    # p_e = []
    # nis = []

    # for subgraph_nodes in tqdm(community_groups):
    #     if len(subgraph_nodes) < 10:
    #         continue
    #     node_ids = torch.tensor(subgraph_nodes)
    #     n_i = len(node_ids)
    #     # A_s = A[node_ids, node_ids]

    #     # internal_nodes_i = sum(sum(A_s))

    #     # p_s = internal_nodes_i / (len(node_ids) * (len(node_ids) - 1))

    #     external_nodes = all_nodes[~all_nodes.unsqueeze(1).eq(node_ids).any(1)]
    #     edge_mask = edge_index.unsqueeze(2).eq(node_ids).any(2).any(0)
    #     intra_mask = edge_index.unsqueeze(2).eq(node_ids).any(2).all(0)
    #     all_edges = edge_index[:, edge_mask]
    #     intra_edges = edge_index[:, intra_mask]

    #     internal_edges_i = intra_edges.shape[1]
    #     external_edges = all_edges.shape[1] - internal_edges_i
    #     internal_nodes += internal_edges_i

    #     p_s.append(internal_edges_i / (2 * n_i * (n_i - 1)))
    #     p_e.append(external_edges / (2 * n_i * (n - n_i)))
    #     nis.append(n_i)

    # p_s_total = sum([p * n_i for p, n_i in zip(p_s, nis)]) / n
    # p_e_total = sum([p * n_i for p, n_i in zip(p_e, nis)]) / n
    # # inter_connections = edge_index.shape[1] - internal_nodes
    # # p_e = inter_connections / (len(all_nodes) * (len(all_nodes) - 1))

    # print(p_s_total, p_e_total)
    # print(p_s)
    # print(p_e)

    # print(edge_index.shape[1] / (2 * n * (n - 1)))

    # p_s_total, p_e_total = estimate_p_s_p_e(graph)
    # print(p_s_total, p_e_total)
