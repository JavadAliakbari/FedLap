import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Function to generate SBM adjacency matrix
def generate_sbm(n, k, p, q):
    """
    Generate an adjacency matrix for an SBM graph.
    n: Number of nodes.
    k: Number of blocks.
    p: Probability of connection within a block.
    q: Probability of connection between blocks.
    """
    blocks = np.random.choice(range(k), size=n)  # Assign nodes to blocks
    adjacency_matrix = np.random.rand(n, n)  # Generate random matrix

    # Create mask for within-block and between-block connections
    within_block_mask = blocks[:, None] == blocks[None, :]
    between_block_mask = ~within_block_mask

    # Apply probabilities
    adjacency_matrix[within_block_mask] = adjacency_matrix[within_block_mask] < p
    adjacency_matrix[between_block_mask] = adjacency_matrix[between_block_mask] < q

    # Ensure the matrix is symmetric and has no self-loops
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, 1).T

    return adjacency_matrix, blocks


# Function to calculate LLR for a specific edge
def calculate_llr(Y, Q, blocks, u, v, p, q):
    """
    Calculate the log-likelihood ratio (LLR) for edge (u, v).
    Y: Observed matrix.
    Q: Orthonormal basis matrix.
    u, v: Nodes of the edge.
    p, q: SBM probabilities.
    """
    block_u = blocks[u]
    mask = blocks == block_u
    n, m = Q.shape
    Q_u = Q[mask]
    Q_nu = Q[~mask]
    mu_u = p * Q_u.sum(axis=0) + q * Q_nu.sum(axis=0)
    # mu_u = p * Q.sum(axis=0) if u == v else q * Q.sum(axis=0)
    s = np.zeros(n)
    s[mask] = p * (1 - p)
    s[~mask] = q * (1 - q)
    Sigma_u = Q.T @ np.diag(s) @ Q

    # Sigma_u = np.zeros((m, m))
    # for k in range(n):
    #     block_k = blocks[k]
    #     p_uk = p if block_k == block_u else q
    #     Sigma_u += p_uk * (1 - p_uk) * np.outer(Q[k, :], Q[k, :])

    Sigma_u_inv = np.linalg.pinv(Sigma_u)
    Q_v = Q[v, :]

    # Calculate LLR
    term1 = (Y[u, :] - mu_u - 0.5 * Q_v).dot(Sigma_u_inv).dot(Q_v)
    return term1


# Simulate the setup and calculate LLR curves
n = 500  # Number of nodes
k = 4  # Number of blocks
m = 10  # Number of eigenvectors

# Fix p and vary q
p_fixed = 0.4
q_values = np.linspace(0.01, 0.5, 50)
llr_fixed_p = []
u, v = 0, 1  # Fixed edge for analysis
num_experiments = 10000

for q in tqdm(q_values):
    llr_experiments = []
    for _ in range(num_experiments):
        L, blocks = generate_sbm(n, k, p_fixed, q)
        Q = np.random.randn(n, m)
        Q, _ = np.linalg.qr(Q)  # Orthonormalize Q

        Y = L @ Q

        llr = calculate_llr(Y, Q, blocks, u, v, p_fixed, q)
        llr_experiments.append(llr)
    llr_fixed_p.append(np.mean(llr_experiments))

# Fix q and vary p
q_fixed = 0.05
p_values = np.linspace(0.1, 0.9, 50)
llr_fixed_q = []

for p in tqdm(p_values):
    llr_experiments = []
    for _ in range(num_experiments):
        L, blocks = generate_sbm(n, k, p, q_fixed)
        Q = np.random.randn(n, m)
        Q, _ = np.linalg.qr(Q)  # Orthonormalize Q

        Y = L @ Q

        llr = calculate_llr(Y, Q, blocks, u, v, p, q_fixed)
        llr_experiments.append(llr)
    llr_fixed_q.append(np.mean(llr_experiments))

# Plot the LLR values for fixed p and varying q
plt.figure(figsize=(10, 5))
plt.plot(q_values, llr_fixed_p, label=f"Fixed p={p_fixed}", marker="o")
plt.xlabel("q (Between-block connection probability)")
plt.ylabel("LLR")
plt.title("LLR vs q for fixed p")
plt.grid(True)
plt.legend()
# plt.show()

# Plot the LLR values for fixed q and varying p
plt.figure(figsize=(10, 5))
plt.plot(p_values, llr_fixed_q, label=f"Fixed q={q_fixed}", marker="o")
plt.xlabel("p (Within-block connection probability)")
plt.ylabel("LLR")
plt.title("LLR vs p for fixed q")
plt.grid(True)
plt.legend()
plt.show()
