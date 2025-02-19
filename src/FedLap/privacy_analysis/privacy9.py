import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import networkx as nx
from scipy.stats import norm  # Import norm module to use 'pdf' function


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src.FedLap.privacy_analysis.privacy2 import calculate_gaussian_parameters
from src.FedLap.privacy_analysis.test import estimate_p_s_p_e
from src.GNN.Lanczos import arnoldi_iteration
from src import *
from src.utils.define_graph import define_graph


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


m = 100
# u, v = 10, 12  # Fixed edge for analysis

colors = [
    (0, "white"),
    (0.5, "blue"),
    (1, "red"),
    # (-1, "red"),
]  # Replace 'blue' and 'red' with your desired colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

graph = define_graph(config.dataset.dataset_name)
n = graph.num_nodes
edge_index = graph.edge_index
A = create_adj(edge_index, graph.num_nodes).coalesce()
A = reorder_adj(A.to_dense(), graph.edge_index)

A = A.to_sparse()

_, Q = arnoldi_iteration(A, m, log=True)
Q = Q.float()
x = torch.einsum("ij,ij->i", Q, Q)
x = torch.clip(x, 0, 0.1)
values = A.values()
indices = A.indices()

plt.figure(figsize=(16, 9))
h, bins, _ = plt.hist(x, bins=500)
mm = x.mean()
height = max(h)
plt.vlines(mm, 0, height, colors="green", linestyles="dashed", label="mean")
plt.vlines(m / n, 0, height, colors="red", linestyles="dashed", label="m/n")
plt.title("Q histogram")
plt.legend()
plt.savefig(f"Q_histogram_{config.dataset.dataset_name}.png")

# A = A.to_dense().numpy()

blocks = graph.y

p, q = estimate_p_s_p_e(graph)

dbar = 2 * A.values().shape[0] / (n - 1)
pp = dbar / n

llr_matrix = torch.zeros((n, n))
# posterior_probabilities = torch.zeros((n, n))
for u in tqdm(range(n)):
    mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, blocks, p, q, u)
    B_u = torch.einsum("ij,jk->ik", Q, Sigma_u_inv)
    llr_u = torch.einsum("i,ik->k", (-P_u + A[u]), B_u)
    llr = torch.einsum("ij, ij->i", llr_u[None, :] - 0.5 * B_u, Q)
    llr_matrix[u, :] = llr
    # lr = torch.exp(-llr)
    # prior_ratio = (1 - P_u) / P_u
    # posterior_ratio = 1 / (1 + lr * prior_ratio)

    # posterior_probabilities[u, :] = posterior_ratio

# A = A.to_dense()
# llr1 = llr_matrix[A == 1].flatten()
# llr1 = torch.clip(llr1, -20, 1500)
# plt.figure(figsize=(16, 9))
# hist, bins, _ = plt.hist(llr1, bins=2000)
# pos_height = max(hist[:-10])
pos_mean = m / (2 * dbar)
pos_var = 2 * pos_mean
# p_pos = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, pos_mean, np.sqrt(pos_var))
# plt.plot(
#     bins,
#     pos_height * p_pos,
#     "red",
#     linewidth=2,
#     label="Fitted Normal Distribution $L_{uv}=1$",
# )
# plt.title("positive LLR histogram")
# plt.legend()
# plt.xlim(-20, 200)
# plt.ylim(0, pos_height)
# plt.savefig(f"positive_llr_histogram_{config.dataset.dataset_name}.png")

# llr0 = llr_matrix[A == 0].flatten()
# llr0 = torch.clip(llr0, -100, 100)

# plt.figure(figsize=(16, 9))
# hist, bins, _ = plt.hist(llr0, bins=2000)
# neg_height = max(hist[10:-10])
# p_neg = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, -pos_mean, np.sqrt(pos_var))
# plt.plot(
#     bins,
#     neg_height * p_neg,
#     "red",
#     linewidth=2,
#     label="Fitted Normal Distribution $L_{uv}=0$",
# )
# plt.title("negative LLR histogram")
# plt.xlim(-100, 100)
# plt.ylim(0, neg_height)
# plt.legend()
# plt.savefig(f"negative_llr_histogram_{config.dataset.dataset_name}.png")
# del llr0
llr_vector = llr_matrix.flatten()
llr_vector = torch.clip(llr_vector, -100, 100)

plt.figure(figsize=(16, 9))

hist, bins, _ = plt.hist(llr_vector, bins=2000)
max_height = max(hist)
# pdf = p * p_pos + (1 - p) * p_neg
mean = pp * pos_mean + (1 - pp) * -pos_mean
var = (pos_var + pos_mean**2) - mean**2
p_neg = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, -pos_mean, np.sqrt(pos_var))
pdf = np.sqrt(2 * np.pi * var) * norm.pdf(bins, mean, np.sqrt(var))
# pdf
plt.plot(
    bins,
    max_height * pdf,
    "blue",
    linewidth=2,
    label="Mixture of Fitted Normal Distributions",
)
plt.plot(
    bins,
    max_height * p_neg,
    "red",
    linewidth=2,
    label="Fitted Normal Distribution $L_{uv}=0$",
)
plt.xlim(-100, 100)
plt.ylim(0, max_height)
plt.title("LLR histogram")
plt.legend()
plt.savefig(f"llr_histogram_{config.dataset.dataset_name}.png")
