from copy import deepcopy
import os
import sys


pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from src import *
from src.utils.define_graph import define_graph
from src.GNN.Lanczos import arnoldi_iteration


graph = define_graph(config.dataset.dataset_name)
A = create_adj(
    graph.edge_index,
    # normalization="rw",
    self_loop=True,
    num_nodes=graph.num_nodes,
    nodes=graph.node_ids,
)
##########################
n = graph.num_nodes
m = 100
b_true = torch.randn(n, dtype=torch.double)
H_true, Q_true = arnoldi_iteration(A, m, b_true, log=True)

P_true = Q_true @ Q_true.T


A1 = deepcopy(A).to_dense()
A2 = deepcopy(A).to_dense()
rand_ind = np.random.randint(0, graph.edge_index.shape[1], 1)[0]
A2[graph.edge_index[0, rand_ind], graph.edge_index[1, rand_ind]] = 0


D1 = []
D2 = []
for i in tqdm(range(1000)):
    rr = torch.randn(2, dtype=torch.double)
    b = deepcopy(b_true)
    b[graph.edge_index[:, rand_ind]] = rr
    # b[graph.edge_index[1, rand_ind]] = rr[1]

    H1, Q1 = arnoldi_iteration(A1, m, b, log=False)
    # P1 = np.einsum("ij,kj->ik", Q1, Q1)
    P1 = Q1 @ Q1.T
    d1 = torch.norm(P_true - P1, p="fro")
    D1.append(d1)

    H2, Q2 = arnoldi_iteration(A2, m, b, log=False)
    # P2 = np.einsum("ij,kj->ik", Q2, Q2)
    P2 = Q2 @ Q2.T
    d2 = torch.norm(P_true - P2, p="fro")
    D2.append(d2)

plt.hist(D1, bins=100, alpha=0.5, label="D1")
plt.hist(D2, bins=100, alpha=0.5, label="D2")
plt.legend(loc="upper right")
plt.show()
