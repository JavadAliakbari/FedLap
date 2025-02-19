import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import numpy as torch
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src.FedLap.privacy_analysis.test import estimate_p_s_p_e
from src.GNN.Lanczos import arnoldi_iteration, estimate_eigh
from src import *
from src.utils.define_graph import define_graph
from src.FedLap.utils import *


def sigmoid(x, l=1):
    return 1 / (1 + torch.exp(-l * x))


if __name__ == "__main__":
    m = 100

    graph = define_graph(config.dataset.dataset_name)
    n = graph.num_nodes
    blocks = graph.y
    p, q = estimate_p_s_p_e(graph)
    edge_index = graph.edge_index
    A = create_adj(edge_index, graph.num_nodes).coalesce()
    A = reorder_adj(A.to_dense(), graph.edge_index)
    AA = sigmoid(A, l=20)

    plt.figure(figsize=(16, 9))
    plt.imshow(AA, cmap=custom_cmap)
    plt.colorbar()
    plt.title("Adjacency matrix")
    plt.savefig("adjacency_matrix.png")

    b = torch.randn(n)
    H, U = arnoldi_iteration(A, m, b, log=False)
    # D, Q = estimate_eigh(A, m, method="arnoldi", log=False)
    U = U.float()
    H = H.float()

    Ap = torch.einsum("ij,jk,nk->in", U, H, U)
    Ap = torch.clip(Ap, 0, 1)
    Ap = sigmoid(Ap, l=10)

    plt.figure(figsize=(16, 9))
    plt.imshow(Ap, cmap=custom_cmap)
    plt.colorbar()
    plt.title("Arnoldi Truncation matrix")
    plt.savefig("arnoldi_matrix.png")

    D, Q = torch.lobpcg(A, m, largest=True)
    Ap = torch.einsum("ij,j,nj->in", Q, D, Q)
    Ap = torch.clip(Ap, 0, 1)
    Ap = sigmoid(Ap, l=10)

    plt.figure(figsize=(16, 9))
    plt.imshow(Ap, cmap=custom_cmap)
    plt.colorbar()
    plt.title("Specral Truncation matrix")
    plt.savefig("spectral_matrix.png")
