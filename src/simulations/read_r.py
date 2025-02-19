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

    models_ = [
        "flga_S_Arnoldi",
        "flga_DGCN_F+S_Arnoldi",
    ]

    train_ratio_range = [0.1, 0.3, 0.5]

    models = []
    for train_ratio in train_ratio_range:
        for model in models_:
            models.append(f"{model}_{train_ratio}")

    m_range = np.arange(20, 201, 20)
    for m in m_range:
        for partitioning in ["random"]:
            # for partitioning in ["random", "louvain", "kmeans"]:
            if partitioning == "random":
                # num_subgraphs_list = [10]
                num_subgraphs_list = [10]
            else:
                num_subgraphs_list = [10]

            for num_subgraphs in num_subgraphs_list:
                data_ = []
                for train_ratio in train_ratio_range:
                    test_ratio = 0.9 - train_ratio

                    base_path = (
                        "./results/Simulation_change_r3sfgsf/"
                        f"{config.dataset.dataset_name}/"
                    )
                    simulation_path = (
                        f"{base_path}"
                        f"{partitioning}/"
                        f"{m}/"
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
                    df2 = df.loc[models_, "0"].tolist()
                    data = [x.split("±") for x in df2]
                    data_ += [100 * float(x[0]) for x in data]
                plot_data.append(data_)

    df = pd.DataFrame(plot_data, columns=models, index=m_range)
    df.to_csv(f"{base_path}{dataset}_ratio.csv")

    plt.plot(m_range, plot_data, "-*", label=models)
    plt.legend()
    plt.xlabel("training ratio")
    plt.ylabel("accuracy")
    plt.title(f"training ratio vs accuracy for {dataset} {partioning} partitioning")
    plt.show()

# server_GNN
