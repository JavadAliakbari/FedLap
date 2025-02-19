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
    b = torch.randn(graph.num_nodes, dtype=torch.double)
    H1, Q1 = arnoldi_iteration(A1, m, b, log=False)
    AA = Q1 @ H1 @ Q1.T

    d = torch.norm(A1 - AA, p="fro")
    D1.append(d)
    AA[AA > 0.5] = 1
    AA[AA <= 0.5] = 0
    d2 = torch.sum(AA != A1)
    D2.append(d2)

plt.plot(D1)
plt.figure()
plt.plot(D2)
plt.show()
