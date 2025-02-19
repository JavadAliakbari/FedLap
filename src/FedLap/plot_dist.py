import os
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

pythotorchath = os.getcwd()
if pythotorchath not in sys.path:
    sys.path.append(pythotorchath)

from src.FedLap.privacy_analysis.test import estimate_p_s_p_e
from src.utils.define_graph import define_graph
from src.utils.utils import *

graph = define_graph(config.dataset.dataset_name)
n = graph.num_nodes
edge_index = graph.edge_index
A = create_adj(edge_index, graph.num_nodes).coalesce()

A = A.to_sparse()

values = A.values()
indices = A.indices()

# p, q = estimate_p_s_p_e(graph)

dbar = 2 * A.values().shape[0] / (n - 1)
p = dbar / n

# Prepare data for the table
g = np.arange(-5, 15, 0.1)
data = {"Threshold": g}

m_range = np.arange(50, 300, 50)
colors = cm.viridis(
    np.linspace(0, 1, len(m_range))
)  # Generate colors from the viridis colormap
for i, m in enumerate(m_range):
    alpha = m / (p * n)
    x1 = (g - alpha / 2) / np.sqrt(alpha)
    x2 = (g + alpha / 2) / np.sqrt(alpha)
    TPR = 1 - norm.cdf(x1)
    FPR = 1 - norm.cdf(x2)
    pr = p * TPR / (p * TPR + (1 - p) * FPR)

    # Add data to the table
    data[f"TPR_r_{m}"] = TPR
    data[f"FPR_r_{m}"] = FPR
    data[f"Precision_r_{m}"] = pr
    data[f"TPR_Precision_r_{m}"] = TPR + pr

    plt.plot(g, pr + TPR, label=f"TPR + Precision r={m}", color=colors[i])
    plt.plot(g, TPR, "--", label=f"TPR r={m}", color=colors[i])
    plt.plot(g, pr, ":", label=f"Precision r={m}", color=colors[i])

plt.xlabel("Threshold")
plt.ylabel("TPR + Precision")
plt.title("TPR + Precision vs Threshold")
plt.grid(True)
plt.legend()
plt.savefig(f"TPR_precision_threshold_{config.dataset.dataset_name}.png")
# plt.show()

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data)
df.to_csv(f"TPR_precision_data_{config.dataset.dataset_name}.csv", index=False)

m = 100
plt.figure(figsize=(24, 16))
plt.plot(data[f"TPR_r_{m}"], data[f"Precision_r_{m}"])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.grid(True)
plt.savefig(f"precision_recall_{config.dataset.dataset_name}.png")
