import os
import sys
import numpy as torch
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import networkx as nx
from scipy.linalg import solve


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

# from src.FedLap.privacy_analysis.privacy2 import calculate_gaussian_parameters
from src.FedLap.privacy_analysis.test import estimate_p_s_p_e
from src.GNN.Lanczos import arnoldi_iteration
from src import *
from src.utils.define_graph import define_graph


m = 100
# u, v = 10, 12  # Fixed edge for analysis
num_experiments = 100000  # Number of experiments

graph = define_graph(config.dataset.dataset_name)
n = graph.num_nodes
edge_index = graph.edge_index
A = create_adj(edge_index, graph.num_nodes).coalesce()

values = A.values()
indices = A.indices()


def run_experiment(A, m):
    b = torch.randn(n)
    _, Q = arnoldi_iteration(A, m, b, log=True)
    # Q = Q.numpy()
    Q = Q.float()

    blocks = graph.y

    p, q = estimate_p_s_p_e(graph)

    P = torch.where(blocks[:, None] == blocks[None, :], p, q)
    S = P * (1 - P)
    Sig = torch.einsum("ij,ik->ijk", Q, Q)
    Sigma = torch.einsum("ni,ijk->njk", S, Sig)
    Sigma_inv = torch.linalg.inv(Sigma)

    B = torch.einsum("ij,njk->nik", Q, Sigma_inv)
    llr = torch.einsum("ni,nik->nk", (-P + A), B)
    llr_matrix = torch.einsum("nij, ij->ni", llr[:, None, :] - 0.5 * B, Q)
    lr = torch.exp(llr_matrix)
    prior_ratio = P / (1 - P)
    posterior_probabilities = 1 / (1 + lr * prior_ratio)

    llr_curve = []
    for th in np.linspace(-50, 50, 1000):
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
        # FPR = np.sum(negative_llr_experiments > th) / num_experiments
        # TPR = np.sum(positive_llr_experiments > th) / num_experiments
        llr_curve.append((precision, recall))
        # print(np.sum(np.array(positive_llr_experiments) > th) / num_experiments)

    # Posterior probabilities
    posterior_curve = []
    for th in np.linspace(0, 1, 1000):
        mask = posterior_probabilities < th

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
        # FPR = np.sum(negative_llr_experiments > th) / num_experiments
        # TPR = np.sum(positive_llr_experiments > th) / num_experiments
        posterior_curve.append((precision, recall))
        # print(np.sum(np.array(positive_llr_experiments) > th) / num_experiments)

    return llr_curve, posterior_curve, accuracy, precision, recall, f1_score_


llr_curve, posterior_curve, accuracy, precision, recall, f1_score_ = run_experiment(
    A, m
)
# Plot the ROC curve
plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in llr_curve], [x[1] for x in llr_curve])
plt.plot(
    [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
)  # Add diagonal line
plt.xlabel("Precision")
plt.ylabel("Recall")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
plt.title("Recall Precission curve")
# plt.yscale("log")
# plt.xscale("log")
plt.grid(True)
plt.savefig("llr_curve.png")


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score_}")

# Posterior probabilities
# Plot the ROC curve

plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in posterior_curve], [x[1] for x in posterior_curve])
plt.plot(
    [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
)  # Add diagonal line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("ROC curve")
# plt.yscale("log")
# plt.xscale("log")
plt.grid(True)
plt.savefig("posterior_curve.png")


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
