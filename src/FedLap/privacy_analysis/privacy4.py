import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import networkx as nx


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src.FedLap.privacy_analysis.privacy2 import calculate_gaussian_parameters
from src.FedLap.privacy_analysis.test import estimate_p_s_p_e
from src.GNN.Lanczos import arnoldi_iteration
from src import *
from src.utils.define_graph import define_graph

# torch.random.seed(0)
# torch.set_printoptions(suppress=True)


from src.FedLap.utils import *


m = 100
# u, v = 10, 12  # Fixed edge for analysis

graph = define_graph(config.dataset.dataset_name)
n = graph.num_nodes
edge_index = graph.edge_index
A = create_adj(edge_index, graph.num_nodes).coalesce()
A = reorder_adj(A.to_dense(), graph.edge_index)

plt.figure(figsize=(16, 9))
plt.imshow(A, cmap=custom_cmap)
plt.colorbar()
plt.title("Adjacency matrix")
plt.savefig("adjacency_matrix.png")

A = A.to_sparse()

_, Q = arnoldi_iteration(A, m, log=True)
Q = Q.float()
values = A.values()
indices = A.indices()
# A = A.to_dense().numpy()

blocks = graph.y

p, q = estimate_p_s_p_e(graph)


# blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
# Q = torch.random.randn(n, m)
# Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q


llr_matrix = torch.zeros((n, n))
posterior_probabilities = torch.zeros((n, n))
for u in tqdm(range(n)):
    mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, blocks, p, q, u)
    B_u = torch.einsum("ij,jk->ik", Q, Sigma_u_inv)
    llr_u = torch.einsum("i,ik->k", (-P_u + A[u]), B_u)
    llr = torch.einsum("ij, ij->i", llr_u[None, :] - 0.5 * B_u, Q)
    lr = torch.exp(-llr)
    prior_ratio = (1 - P_u) / P_u
    posterior_ratio = 1 / (1 + lr * prior_ratio)

    llr_matrix[u, :] = llr
    posterior_probabilities[u, :] = posterior_ratio

llr_vector = llr_matrix.flatten()
llr_vector = torch.clip(llr_vector, -100, 100)
plt.figure(figsize=(16, 9))
plt.hist(llr_vector, bins=1000)
plt.title("LLR histogram")
plt.savefig("llr_histogram2.png")


llr_matrix2 = torch.clip(llr_matrix, -20, 20)
# llr_matrix2 = llr_matrix2 - llr_matrix2.min()
plt.figure(figsize=(16, 9))
plt.imshow(llr_matrix2, cmap=custom_cmap)
plt.colorbar()
plt.title("LLR matrix")
plt.savefig("llr_matrix.png")

plt.figure(figsize=(16, 9))
plt.imshow(posterior_probabilities, cmap=custom_cmap)
plt.colorbar()
plt.title("Posterior probabilities")
plt.savefig("posterior_probabilities.png")

roc_curve = []
pr_curve = []  # Precision-recall curve
for th in tqdm(torch.linspace(-500, 500, 1000)):
    mask = llr_matrix > th

    # Select values from sparse matrix using the mask
    selected_values = values[mask[indices[0], indices[1]]]

    # Sum the selected values
    true_positive = selected_values.sum()
    false_positive = torch.sum(mask) - true_positive

    selected_values2 = values[~mask[indices[0], indices[1]]]
    false_negative = selected_values2.sum()
    true_negative = torch.sum(~mask) - false_negative

    FPR = false_positive / (false_positive + true_negative)
    TPR = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / n**2
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score_ = 2 * precision * recall / (precision + recall)
    # estimate_A[llr_matrix < 0] = 0
    # FPR = torch.sum(negative_llr_experiments > th) / num_experiments
    # TPR = torch.sum(positive_llr_experiments > th) / num_experiments
    roc_curve.append((FPR, TPR))
    pr_curve.append((recall, precision))

    # print(torch.sum(torch.array(positive_llr_experiments) > th) / num_experiments)

# Plot the ROC curve
plt.figure(figsize=(10, 5))
fig, axs = plt.subplots(1, 2)
axs[0].plot([x[0] for x in roc_curve], [x[1] for x in roc_curve])
axs[0].plot(
    [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
)  # Add diagonal line
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
# axs[0].set_grid(True)
axs[1].plot([x[0] for x in pr_curve], [x[1] for x in pr_curve])
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precission")
# axs[1].set_grid(True)
plt.title("ROC curve")
plt.savefig("roc_curve2.png")


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score_}")
print(f"FPR: {FPR}")
print(f"TPR: {TPR}")

# Posterior probabilities
roc_curve = []
pr_curve = []
for th in tqdm(torch.linspace(0, 1, 1000)):
    mask = posterior_probabilities > th

    # Select values from sparse matrix using the mask
    selected_values = values[mask[indices[0], indices[1]]]

    # Sum the selected values
    true_positive = selected_values.sum()
    false_positive = torch.sum(mask) - true_positive

    selected_values2 = values[~mask[indices[0], indices[1]]]
    false_negative = selected_values2.sum()
    true_negative = torch.sum(~mask) - false_negative

    accuracy = (true_positive + true_negative) / n**2
    precision = true_positive / (true_positive + false_positive)
    FPR = false_positive / (false_positive + true_negative)
    TPR = true_positive / (true_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    f1_score_ = 2 * precision * recall / (precision + recall)
    # estimate_A[llr_matrix < 0] = 0
    # FPR = torch.sum(negative_llr_experiments > th) / num_experiments
    # TPR = torch.sum(positive_llr_experiments > th) / num_experiments
    roc_curve.append((FPR, TPR))
    pr_curve.append((recall, precision))
    # print(torch.sum(torch.array(positive_llr_experiments) > th) / num_experiments)


# Plot the ROC curve
plt.figure(figsize=(10, 5))
fig, axs = plt.subplots(1, 2)
axs[0].plot([x[0] for x in roc_curve], [x[1] for x in roc_curve])
axs[0].plot(
    [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
)  # Add diagonal line
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
# axs[0].grid(True)
axs[1].plot([x[0] for x in pr_curve], [x[1] for x in pr_curve])
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
# axs[1].grid(True)
plt.title("ROC curve")
plt.savefig("roc_curve3.png")


# estimate_A_poterior = torch.zeros((n, n))
# estimate_A_poterior[posterior_probabilities > 0.5] = 1
# # estimate_A_poterior[posterior_probabilities < P] = 0

# true_positive = torch.sum(torch.logical_and(estimate_A_poterior == 1, A == 1))
# false_positive = torch.sum(torch.logical_and(estimate_A_poterior == 1, A == 0))

# true_negative = torch.sum(torch.logical_and(estimate_A_poterior == 0, A == 0))
# false_negative = torch.sum(torch.logical_and(estimate_A_poterior == 0, A == 1))

# accuracy = (true_positive + true_negative) / n**2
# precision = true_positive / (true_positive + false_positive)
# recall = true_positive / (true_positive + false_negative)
# f1_score_ = 2 * precision * recall / (precision + recall)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score_}")
print(f"FPR: {FPR}")
print(f"TPR: {TPR}")
