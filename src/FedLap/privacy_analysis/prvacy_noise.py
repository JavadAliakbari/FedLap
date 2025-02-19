import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)


# Function to generate SBM adjacency matrix
def generate_sbm(n, p, d):
    """
    Generate an adjacency matrix for an SBM graph.
    n: Number of nodes.
    p: Probability of connection within a block.
    """
    adjacency_matrix = torch.rand(d, n)  # Generate random matrix

    # Apply probabilities
    adjacency_matrix = (adjacency_matrix < p).float()

    return adjacency_matrix


# Function to calculate LLR for a specific edge
def calculate_llr(Y_u, Q_v, mu_u, Sigma_u_inv):
    """
    Calculate the log-likelihood ratio (LLR) for edge (u, v).
    Y: Observed matrix.
    Q: Orthonormal basis matrix.
    u, v: Nodes of the edge.
    p, q: SBM probabilities.
    """

    # Calculate LLR
    term1 = torch.einsum("ij,jk,kn->i", (Y_u - mu_u - 0.5 * Q_v), Sigma_u_inv, Q_v)
    return term1


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


if __name__ == "__main__":
    # Simulate the setup and calculate LLR curves
    n = 10000  # Number of nodes
    k = 10  # Number of blocks
    m = 150  # Number of eigenvectors

    p = 0.02
    node_per_block = n // k
    dbar = n * p
    u, v = 10, 12  # Fixed edge for analysis
    num_experiments = 200  # Number of experiments
    s = 0.3

    therotical_threshold = m / (2 * dbar) - m * np.log(1 + 1 / dbar) / (2)

    # blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
    Q = torch.randn(n, m)
    Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q
    noise = s * torch.randn(n, m)
    Q_n = Q + noise
    Q_n = Q_n / torch.linalg.norm(Q_n, dim=1)[:, None]
    Q_v = Q_n[v, :]
    e_v = torch.zeros(n)
    e_v[v] = 1
    e_v = e_v[None, :]

    mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q_n, p)
    B = torch.einsum("ij,j->i", Q_n, Sigma_u_inv @ Q_v)

    P_u_ = P_u[None, :]
    mu_u = mu_u[None, :]
    Q_v = Q_v[None, :]

    positive_llr_experiments = []
    positive_posterior_experiments = []
    negative_llr_experiments = []
    negative_posterior_experiments = []
    d = 10000
    for _ in tqdm(range(num_experiments)):
        L_u = generate_sbm(n, p, d)
        # L_u = L[u, :]
        L_u[:, v] = 0

        Y = L_u @ Q
        # llr_n = (L_u - P_u - 0.5 * e_v) @ B @ e_v
        # llr_n = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        llr_n = calculate_llr(Y, Q_v, mu_u, Sigma_u_inv)
        lr_n = torch.exp(llr_n)
        prior_ratio = P_u[v] / (1 - P_u[v])
        posterior_ratio_n = 1 / (1 + lr_n * prior_ratio)

        negative_llr_experiments.append(llr_n)
        negative_posterior_experiments.append(posterior_ratio_n)
        L_u[:, v] = 1
        Y += Q_v

        # llr_p = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        llr_p = calculate_llr(Y, Q_v, mu_u, Sigma_u_inv)
        lr_p = torch.exp(-llr_p)
        prior_ratio = (1 - P_u[v]) / P_u[v]
        posterior_ratio_p = 1 / (1 + lr_p * prior_ratio)

        # llr_p = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        positive_llr_experiments.append(llr_p)
        positive_posterior_experiments.append(posterior_ratio_p)

    positive_llr_experiments, negative_llr_experiments = (
        torch.stack(positive_llr_experiments).flatten(),
        torch.stack(negative_llr_experiments).flatten(),
    )
    positive_posterior_experiments, negative_posterior_experiments = (
        torch.stack(positive_posterior_experiments).flatten(),
        torch.stack(negative_posterior_experiments).flatten(),
    )
    # Calculate ROC curve
    roc_curve = []
    pr_curve = []
    for th in torch.linspace(-20, 20, 1000):
        FPR = (
            torch.sum(negative_llr_experiments > th) / negative_llr_experiments.shape[0]
        )
        TPR = (
            torch.sum(positive_llr_experiments > th) / negative_llr_experiments.shape[0]
        )
        TP = torch.sum(positive_llr_experiments > th)
        FP = torch.sum(negative_llr_experiments > th)
        TN = torch.sum(negative_llr_experiments < th)
        FN = torch.sum(positive_llr_experiments < th)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        roc_curve.append((FPR, TPR))
        pr_curve.append((recall, precision))
        # print(torch.sum(torch.array(positive_llr_experiments) > th) / num_experiments)

    # Plot the ROC curve
    plt.figure(figsize=(24, 16))
    plt.plot([x[0] for x in roc_curve], [x[1] for x in roc_curve])
    plt.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess"
    )  # Add diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("roc_curve.png")
    # plt.show()

    # Plot the LLR values for fixed q and varying p
    plt.figure(figsize=(24, 16))
    negative_hist, negative_bins, _ = plt.hist(
        negative_llr_experiments, label="$L_{uv}=0$", bins=1000, alpha=0.5
    )
    positive_hist, positive_bins, _ = plt.hist(
        positive_llr_experiments, label="$L_{uv}=1$", bins=1000, alpha=0.5
    )
    max_height = max(max(negative_hist), max(positive_hist))
    plt.vlines(0, 0, max_height, color="red", label="Threshold")
    plt.vlines(
        therotical_threshold,
        0,
        max_height,
        color="green",
        label="Theoretical threshold",
    )
    plt.ylabel("Frequency")
    plt.xlabel("LLR")
    plt.title("LLR histogram")
    plt.grid(True)
    plt.legend()
    plt.savefig("llr_histogram.png")

    # Plot the posterior values
    plt.figure(figsize=(24, 16))
    negative_posterior_hist, negative_posterior_bins, _ = plt.hist(
        negative_posterior_experiments, label="$L_{uv}=0$", bins=1000, alpha=0.5
    )
    max_posterior_height = max(negative_posterior_hist)
    x_min = min(negative_posterior_bins)
    plt.vlines(
        torch.mean(negative_posterior_experiments),
        0,
        max_posterior_height,
        color="red",
        linestyle="--",
        label="mean",
    )
    plt.vlines(
        1 - P_u[v],
        0,
        max_posterior_height,
        linestyle="-",
        color="blue",
        label="prior",
    )
    plt.ylabel("Frequency")
    plt.xlabel("posterior")
    plt.xlim(0.9, 1)
    plt.title("posterior histogram")
    plt.grid(True)
    plt.legend()
    plt.savefig("negative_posterior_histogram.png")
    plt.figure(figsize=(24, 16))
    positive_posterior_hist, positive_posterior_bins, _ = plt.hist(
        positive_posterior_experiments, label="$L_{uv}=1$", bins=1000, alpha=0.5
    )
    max_posterior_height = max(positive_posterior_hist)
    x_max = max(positive_posterior_bins)
    plt.vlines(
        torch.mean(positive_posterior_experiments),
        0,
        max_posterior_height,
        color="red",
        linestyle="--",
        label="mean",
    )
    plt.vlines(
        P_u[v],
        0,
        max_posterior_height,
        linestyle="-",
        color="blue",
        label="prior",
    )
    plt.ylabel("Frequency")
    plt.xlabel("posterior")
    plt.xlim(0, 0.1)
    plt.title("posterior histogram")
    plt.grid(True)
    plt.legend()
    plt.savefig("positive_posterior_histogram.png")
    # plt.show()
