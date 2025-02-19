import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import networkx as nx

pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src.GNN.Lanczos import arnoldi_iteration

# torch.random.seed(0)
# torch.set_printoptions(suppress=True)


# Function to generate SBM adjacency matrix
def generate_sbm(blocks, u, p, q, d):
    """
    Generate an adjacency matrix for an SBM graph.
    n: Number of nodes.
    k: Number of blocks.
    p: Probability of connection within a block.
    q: Probability of connection between blocks.
    """
    n = blocks.shape[0]
    adjacency_matrix = torch.rand(d, n)  # Generate random matrix

    # Create mask for within-block and between-block connections
    within_block_mask = blocks == blocks[u]
    between_block_mask = ~within_block_mask

    # Apply probabilities
    adjacency_matrix[:, within_block_mask] = (
        adjacency_matrix[:, within_block_mask] < p
    ).float()
    adjacency_matrix[:, between_block_mask] = (
        adjacency_matrix[:, between_block_mask] < q
    ).float()

    # Ensure the matrix is symmetric and has no self-loops
    # torch.fill_diagonal(adjacency_matrix, 0)
    # adjacency_matrix = torch.triu(adjacency_matrix) + torch.triu(adjacency_matrix, 1).T

    return adjacency_matrix


def calculate_gaussian_parameters(Q, blocks, p, q, u):
    block_u = blocks[u]
    mask = blocks == block_u
    n, m = Q.shape
    P_u = torch.zeros(n)
    P_u[mask] = p
    P_u[~mask] = q
    mu_u = torch.einsum("i,ij->j", P_u, Q)
    # mu_u = p * Q.sum(axis=0) if u == v else q * Q.sum(axis=0)
    s = P_u * (1 - P_u)
    # Sigma_u = Q.T @ torch.diag(s) @ Q
    Sigma_u = torch.einsum("i,ij,ik->jk", s, Q, Q)  # Q.T @ torch.diag(s) @ Q
    Sigma_u_inv = torch.linalg.inv(Sigma_u)
    # Sigma_u_inv = torch.linalg.pinv(Sigma_u)
    # B = Q @ Sigma_u_inv @ Q.T

    return mu_u, Sigma_u_inv, P_u


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
    term1 = (Y_u - mu_u - 0.5 * Q_v).dot(Sigma_u_inv).dot(Q_v)
    return term1


if __name__ == "__main__":
    # Simulate the setup and calculate LLR curves
    n = 20000  # Number of nodes
    k = 10  # Number of blocks
    m = 100  # Number of eigenvectors

    # Fix p and vary q
    # Amazon Photo parameters
    # p = 0.012
    # q = 0.00084
    # Cora parameters
    # p = 0.0041
    # q = 0.00033
    p = 0.01
    q = 0.001
    node_per_block = n // k
    dbar = node_per_block * p + (k - 1) * node_per_block * q
    u, v = 10, 12  # Fixed edge for analysis
    num_experiments = 100  # Number of experiments

    # Generate SBM graph
    probs = np.full((k, k), q)
    np.fill_diagonal(probs, p)
    sizes = np.full(k, n // k)
    n = sum(sizes)
    g = nx.stochastic_block_model(sizes, probs, seed=0, sparse=True)
    A = nx.adjacency_matrix(g)
    # A = torch.sparse.from_scipy(g.adjacency_matrix()).coalesce()
    # Convert to PyTorch sparse tensor
    A = torch.sparse_coo_tensor(
        np.array(A.nonzero()),
        A.data,
        A.shape,
    )
    # A, blocks = generate_sbm_true(n, k, p, q)
    # A = torch.tensor(A, dtype=torch.double)
    _, Q = arnoldi_iteration(A, m, log=True)
    Q = Q.float()
    del A
    blocks = torch.tensor([g.nodes[u]["block"] for u in g.nodes])

    # blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
    # Q = torch.random.randn(n, m)
    # Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q
    Q_v = Q[v, :]
    e_v = torch.zeros(n)
    e_v[v] = 1
    e_v = e_v[None, :]

    mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, blocks, p, q, u)
    B = torch.einsum("ij,j->i", Q, Sigma_u_inv @ Q_v)

    P_u_ = P_u[None, :]

    positive_llr_experiments = []
    positive_posterior_experiments = []
    negative_llr_experiments = []
    negative_posterior_experiments = []
    d = 10000
    for _ in tqdm(range(num_experiments)):
        L_u = generate_sbm(blocks, u, p, q, d)
        # L_u = L[u, :]
        L_u[:, v] = 0

        # Y = L_u @ Q
        # llr_n = (L_u - P_u - 0.5 * e_v) @ B @ e_v
        llr_n = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        lr_n = torch.exp(llr_n)
        prior_ratio = P_u[v] / (1 - P_u[v])
        posterior_ratio_n = 1 / (1 + lr_n * prior_ratio)

        # llr_n2 = calculate_llr(Y, Q_v, mu_u, Sigma_u_inv)
        negative_llr_experiments.append(llr_n)
        negative_posterior_experiments.append(posterior_ratio_n)
        L_u[:, v] = 1
        # Y += Q_v

        llr_p = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        lr_p = torch.exp(-llr_p)
        prior_ratio = (1 - P_u[v]) / P_u[v]
        posterior_ratio_p = 1 / (1 + lr_p * prior_ratio)

        # llr_p = torch.einsum("ki,i->k", (L_u - P_u_ - 0.5 * e_v), B)
        # llr_p2 = calculate_llr(Y, Q_v, mu_u, Sigma_u_inv)
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
        m / (2 * dbar), 0, max_height, color="green", label="Theoretical threshold"
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
