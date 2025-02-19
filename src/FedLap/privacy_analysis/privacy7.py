import os
import sys
import numpy as torch
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


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


def calc_roc_curve(A, M, th_range):
    n = A.size(0)
    values = A.values()
    indices = A.indices()
    # Posterior probabilities
    roc_curve = []
    pr_curve = []
    for th in tqdm(th_range, leave=False):  # Threshold
        mask = M > th

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

    return roc_curve, pr_curve


def run_experiment(A, m, p, q, blocks):
    # blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
    # Q = torch.random.randn(n, m)
    # Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q

    n = A.size(0)
    b = torch.randn(n)
    _, Q = arnoldi_iteration(A, m, b, log=False)
    Q = Q.float()

    llr_matrix = torch.zeros((n, n))
    posterior_probabilities = torch.zeros((n, n))
    for u in tqdm(range(n), leave=False):
        mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, blocks, p, q, u)
        B_u = torch.einsum("ij,jk->ik", Q, Sigma_u_inv)
        llr_u = torch.einsum("i,ik->k", (-P_u + A[u]), B_u)
        llr = torch.einsum("ij, ij->i", llr_u[None, :] - 0.5 * B_u, Q)
        lr = torch.exp(-llr)
        prior_ratio = (1 - P_u) / P_u
        posterior_ratio = 1 / (1 + lr * prior_ratio)

        llr_matrix[u, :] = llr
        posterior_probabilities[u, :] = posterior_ratio

    return llr_matrix, posterior_probabilities


if __name__ == "__main__":
    m = 100
    # u, v = 10, 12  # Fixed edge for analysis
    num_experiments = 100000  # Number of experiments

    graph = define_graph(config.dataset.dataset_name)
    n = graph.num_nodes
    blocks = graph.y
    p, q = estimate_p_s_p_e(graph)
    edge_index = graph.edge_index
    A = create_adj(edge_index, graph.num_nodes).coalesce()
    # A = A.to_dense().numpy()

    llr_roc_corves = []
    llr_pr_curves = []
    posterior_roc_curves = []
    posterior_pr_curves = []
    for _ in tqdm(range(10)):
        llr_matrix, posterior_probabilities = run_experiment(A, m, p, q, blocks)
        llr_roc_corve, llr_pr_curve = calc_roc_curve(
            A, llr_matrix, torch.linspace(-50, 50, 250)
        )
        posterior_roc_curve, posterior_pr_curve = calc_roc_curve(
            A, posterior_probabilities, torch.linspace(0, 1, 250)
        )
        llr_roc_corves.append(torch.tensor(llr_roc_corve))
        llr_pr_curves.append(torch.tensor(llr_pr_curve))
        posterior_roc_curves.append(torch.tensor(posterior_roc_curve))
        posterior_pr_curves.append(torch.tensor(posterior_pr_curve))

    llr_roc_corve = torch.stack(llr_roc_corves).mean(dim=0)
    llr_pr_curve = torch.stack(llr_pr_curves).mean(dim=0)
    posterior_roc_curve = torch.stack(posterior_roc_curves).mean(dim=0)
    posterior_pr_curve = torch.stack(posterior_pr_curves).mean(dim=0)

    # Plot the ROC curve
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(1, 2)
    axs[0].plot([x[0] for x in llr_roc_corve], [x[1] for x in llr_roc_corve])
    axs[0].fill_between(
        [x[0] for x in llr_roc_corve],
        [x[1] for x in llr_roc_corve],
        color="skyblue",
        alpha=0.4,
    )
    axs[0].plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
    )  # Add diagonal line
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    # axs[0].set_grid(True)
    axs[1].plot([x[0] for x in llr_pr_curve], [x[1] for x in llr_pr_curve])
    axs[1].fill_between(
        [x[0] for x in llr_pr_curve],
        [x[1] for x in llr_pr_curve],
        color="lightgreen",
        alpha=0.4,
    )
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precission")
    # axs[1].set_grid(True)
    plt.title("ROC curve")
    plt.savefig("roc_curve2.png")

    # Plot the ROC curve
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(
        [x[0] for x in posterior_roc_curve], [x[1] for x in posterior_roc_curve]
    )
    axs[0].fill_between(
        [x[0] for x in posterior_roc_curve],
        [x[1] for x in posterior_roc_curve],
        color="lightgreen",
        alpha=0.4,
    )
    axs[0].plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
    )  # Add diagonal line
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    # axs[0].grid(True)
    axs[1].plot([x[0] for x in posterior_pr_curve], [x[1] for x in posterior_pr_curve])
    axs[1].fill_between(
        [x[0] for x in posterior_pr_curve],
        [x[1] for x in posterior_pr_curve],
        color="lightgreen",
        alpha=0.4,
    )
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    # axs[1].grid(True)
    plt.title("ROC curve")
    plt.savefig("roc_curve3.png")
