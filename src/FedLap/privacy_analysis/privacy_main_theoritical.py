import os
import sys
import numpy as torch
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy.stats import norm  # Import norm module to use 'pdf' function


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src import *

# torch.random.seed(0)
# torch.set_printoptions(suppress=True)


def calculate_gaussian_parameters(Q, p):
    P_u = torch.full((n,), p)
    mu_u = torch.einsum("i,ij->j", P_u, Q)
    # mu_u = p * Q.sum(axis=0) if u == v else q * Q.sum(axis=0)
    s = P_u * (1 - P_u)
    # Sigma_u = Q.T @ torch.diag(s) @ Q
    Sigma_u = torch.einsum("i,ij,ik->jk", s, Q, Q)  # Q.T @ torch.diag(s) @ Q
    Sigma_u_inv = torch.linalg.inv(Sigma_u)
    # Sigma_u_inv = torch.linalg.pinv(Sigma_u)
    # B = Q @ Sigma_u_inv @ Q.T

    return mu_u, Sigma_u_inv, P_u


# Function to generate SBM adjacency matrix
def generate_sbm(n, p):
    """
    Generate an adjacency matrix for an SBM graph.
    n: Number of nodes.
    p: Probability of connection within a block.
    """
    adjacency_matrix = torch.rand(n, n)  # Generate random matrix

    # Apply probabilities
    mask = adjacency_matrix < p
    adjacency_matrix = mask.float()
    # adjacency_matrix = (~(adjacency_matrix > p)).float()

    return adjacency_matrix


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


def run_experiment(A, m, p):
    # blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
    n = A.size(0)
    Q = torch.randn(n, m)
    Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q

    # b = torch.randn(n)
    # _, Q = arnoldi_iteration(A, m, b, log=False)
    # Q = Q.float()

    llr_matrix = torch.zeros((n, n))
    # posterior_probabilities = torch.zeros((n, n))
    mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, p)
    for u in tqdm(range(n), leave=False):
        B_u = torch.einsum("ij,jk->ik", Q, Sigma_u_inv)
        llr_u = torch.einsum("i,ik->k", (-P_u + A[u]), B_u)
        llr = torch.einsum("ij, ij->i", llr_u[None, :] - 0.5 * B_u, Q)
        llr_matrix[u, :] = llr

        # lr = torch.exp(-llr)
        # prior_ratio = (1 - P_u) / P_u
        # posterior_ratio = 1 / (1 + lr * prior_ratio)

        # posterior_probabilities[u, :] = posterior_ratio

    return llr_matrix


def plot_th(p, m, n):
    dbar = n * p

    alpha = m / dbar
    th_range = torch.linspace(-20, 20, 250)
    x1 = (th_range + alpha / 2) / np.sqrt(alpha)
    x2 = (th_range - alpha / 2) / np.sqrt(alpha)

    f1 = p / (np.sqrt(2 * np.pi) * (x1 + np.sqrt(x1**2 + 4)))
    f2 = (1 - p) * np.exp(-th_range) / (np.sqrt(2 * np.pi) * (x2 + np.sqrt(x2**2 + 4)))
    y1 = (p / 2) / (f1 + f2) + 0.5 * np.exp(-(x1**2) / 2)

    return y1


if __name__ == "__main__":
    m = 100
    n = 2000
    p = 0.01
    dbar = n * p
    A = generate_sbm(n, p)

    alpha = m / dbar

    c = np.exp(-alpha / 2)

    llr_pr_curves = []
    llr_matrix = run_experiment(A, m, p)
    th_range = torch.linspace(-20, 20, 250)
    llr_roc_corve, llr_pr_curve = calc_roc_curve(A.to_sparse(), llr_matrix, th_range)

    llr1 = llr_matrix[A == 1].flatten()
    plt.figure(figsize=(16, 9))
    pos_hist, pos_bins, _ = plt.hist(
        llr1, bins=100, density=True, alpha=0.6, color="b", label="LLR $L_{uv}=1$"
    )
    pos_height = max(pos_hist)
    pos_mean = alpha / 2
    pos_var = alpha
    p_pos = norm.pdf(pos_bins, pos_mean, np.sqrt(pos_var))
    # plt.title("positive LLR histogram")
    # plt.legend()
    # plt.savefig("positive_llr_histogram2.png")
    llr0 = llr_matrix[A == 0].flatten()

    # plt.figure(figsize=(16, 9))
    neg_hist, neg_bins, _ = plt.hist(
        llr0, bins=100, density=True, alpha=0.6, color="r", label="LLR $L_{uv}=0$"
    )
    neg_height = max(neg_hist)
    p_neg = norm.pdf(neg_bins, -pos_mean, np.sqrt(pos_var))
    plt.plot(
        pos_bins,
        p_pos,
        "blue",
        linewidth=2,
        label="Fitted Normal Distribution $L_{uv}=1$",
    )
    plt.plot(
        neg_bins,
        p_neg,
        "red",
        linewidth=2,
        label="Fitted Normal Distribution $L_{uv}=0$",
    )
    normalize_factor = np.sqrt(2 * np.pi * pos_var)
    plt.plot(th_range, [x[0] / normalize_factor for x in llr_pr_curve], label="recall")
    plt.plot(
        th_range, [x[1] / normalize_factor for x in llr_pr_curve], label="precision"
    )
    plt.title("LLR histogram")
    plt.legend()
    plt.xlabel("LLR")
    # plt.ylabel("Density")
    plt.savefig("negative_llr_histogram2.png")

    # Plot the ROC curve

    y1 = plot_th(p, m, n)

    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in llr_pr_curve], [x[1] for x in llr_pr_curve])
    plt.plot(th_range, y1, label="Theoretical")
    plt.fill_between(
        [x[0] for x in llr_pr_curve],
        [x[1] for x in llr_pr_curve],
        color="lightgreen",
        alpha=0.4,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precission")
    # plt..set_grid(True)
    plt.title("ROC curve")
    plt.savefig("roc_curve22.png")
