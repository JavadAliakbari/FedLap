import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import norm  # Import norm module to use 'pdf' function


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


m = 100
n = 2000
p = 0.01
dbar = n * p * (1 - p)
A = generate_sbm(n, p)


# blocks = torch.random.choice(range(k), size=n)  # Assign nodes to blocks
Q = torch.randn(n, m)
Q, _ = torch.linalg.qr(Q)  # Orthonormalize Q


llr_matrix = torch.zeros((n, n))
# posterior_probabilities = torch.zeros((n, n))
mu_u, Sigma_u_inv, P_u = calculate_gaussian_parameters(Q, p)
for u in tqdm(range(n)):
    B_u = torch.einsum("ij,jk->ik", Q, Sigma_u_inv)
    llr_u = torch.einsum("i,ik->k", (-P_u + A[u]), B_u)
    llr = torch.einsum("ij, ij->i", llr_u[None, :] - 0.5 * B_u, Q)
    llr_matrix[u, :] = llr
    # lr = torch.exp(-llr)
    # prior_ratio = (1 - P_u) / P_u
    # posterior_ratio = 1 / (1 + lr * prior_ratio)

    # posterior_probabilities[u, :] = posterior_ratio

llr1 = llr_matrix[A == 1].flatten()
plt.figure(figsize=(16, 9))
hist, bins, _ = plt.hist(llr1, bins=1000)
pos_height = max(hist)
pos_mean = m / (2 * dbar)
pos_var = 2 * pos_mean
p_pos = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, pos_mean, np.sqrt(pos_var))
plt.plot(
    bins,
    pos_height * p_pos,
    "red",
    linewidth=2,
    label="Fitted Normal Distribution $L_{uv}=1$",
)
plt.title("positive LLR histogram")
plt.legend()
plt.savefig("positive_llr_histogram2.png")
llr0 = llr_matrix[A == 0].flatten()

plt.figure(figsize=(16, 9))
hist, bins, _ = plt.hist(llr0, bins=1000)
neg_height = max(hist)
p_neg = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, -pos_mean, np.sqrt(pos_var))
plt.plot(
    bins,
    neg_height * p_neg,
    "red",
    linewidth=2,
    label="Fitted Normal Distribution $L_{uv}=0$",
)
plt.title("negative LLR histogram")
plt.legend()
plt.savefig("negative_llr_histogram2.png")
llr_vector = llr_matrix.flatten()

plt.figure(figsize=(16, 9))
hist, bins, _ = plt.hist(llr_vector, bins=1000)
p_neg = np.sqrt(2 * np.pi * pos_var) * norm.pdf(bins, -pos_mean, np.sqrt(pos_var))
max_height = max(hist)
# pdf = p * p_pos + (1 - p) * p_neg
mean = p * pos_mean + (1 - p) * -pos_mean
var = p * (pos_var + pos_mean**2) + (1 - p) * (pos_var + pos_mean**2) - mean**2
pdf = np.sqrt(2 * np.pi * var) * norm.pdf(bins, mean, np.sqrt(var))
# pdf
plt.plot(
    bins,
    max_height * pdf,
    "green",
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
plt.legend()
plt.title("LLR histogram")
plt.savefig("llr_histogram2.png")
