from os import listdir
import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.utils import *

if __name__ == "__main__":
    # folder_path = "results/Paper Results/Cora/louvain-10/0.1/"
    # dataset = "Chameleon"
    # partioning = "kmeans"
    dataset = "Cora"
    partioning = "random"
    num_subgraphs = 10
    plot_data = []

    models = [
        "fedpub",
        "fedgcn2",
        "flga_GNN",
        "flga_S_Laplacian",
        "flga_S_Arnoldi",
        "flga_DGCN_F+S_DGCN",
        "flga_DGCN_F+S_Laplacian",
        "flga_DGCN_F+S_Arnoldi",
    ]

    for partitioning in ["random"]:
        # for partitioning in ["random", "louvain", "kmeans"]:
        num_subgraphs_list = np.arange(5, 55, 5)

        for num_subgraphs in num_subgraphs_list:
            data_ = []
            for train_ratio in [config.subgraph.train_ratio]:
                test_ratio = config.subgraph.test_ratio

                base_path = (
                    "results/Simulation_clients/" f"{config.dataset.dataset_name}/"
                )
                simulation_path = (
                    f"{base_path}"
                    f"{partitioning}/"
                    f"{num_subgraphs}/"
                    f"{train_ratio:0.2f}/"
                )
                filenames = listdir(simulation_path)
                filename = [
                    filename for filename in filenames if filename.endswith(".csv")
                ][0]
                # folder_path = f"ICML Results/train_ratio/{dataset}/{partioning}/{num_subgraphs}/{train_ratio}/"
                path = f"{simulation_path}{filename}"

                df = pd.read_csv(path, index_col="Unnamed: 0")
                df2 = df.loc[models, "0"].tolist()
                data = [x.split("±") for x in df2]
                data_ += [100 * float(x[0]) for x in data]
            plot_data.append(data_)

    df = pd.DataFrame(plot_data, columns=models, index=num_subgraphs_list)
    df.to_csv(f"{base_path}{dataset}_clients.csv")

    plt.plot(num_subgraphs_list, plot_data, "-*", label=models)
    plt.legend()
    plt.xlabel("number of clients")
    plt.ylabel("accuracy")
    plt.title(f"num of clients vs accuracy for {dataset} {partioning} partitioning")
    plt.show()

# server_GNN
