import os
import sys
import json


pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *

# from src.simulations.configuration import get_configurations
from src.simulations.simulation_utils import (
    calc_average_std_result,
    save_average_result,
)
from src.utils.define_graph import define_graph
from src.utils.logger import getLOGGER
from src.GNN.GNN_server import GNNServer
from src.utils.graph_partitioning import partition_graph


def get_configurations(GNN_server: GNNServer, spectral_epoch):
    configurations = {
        "flga_S_Arnoldi": {
            "method": GNN_server.joint_train_g,
            "epochs": spectral_epoch,
            "data_type": "structure",
            "f_type": None,
            "s_type": "LanczosLaplace",
            "FL": True,
        },
        "flga_DGCN_F+S_Arnoldi": {
            "method": GNN_server.joint_train_g,
            "epochs": spectral_epoch,
            "data_type": "f+s",
            "f_type": "DGCN",
            "s_type": "LanczosLaplace",
            "FL": True,
        },
    }

    return configurations


def create_clients(
    graph, GNN_server: GNNServer, train_ratio, test_ratio, num_subgraphs, partitioning
):
    graph.add_masks(train_ratio=train_ratio, test_ratio=test_ratio)

    GNN_server.reset_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)


def get_GNN_results2(
    GNN_server: GNNServer,
    bar: tqdm,
    spectral_epoch=config.model.iterations,
):
    results = {}

    # res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    # results[f"FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    # results[f"FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    # results[f"FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    configurations = get_configurations(GNN_server, spectral_epoch)
    for name, config in configurations.items():
        res = config["method"](
            epochs=config["epochs"],
            data_type=config["data_type"],
            fmodel_type=config["f_type"],
            smodel_type=config["s_type"],
            FL=config["FL"],
            log=False,
            plot=False,
        )
        results[name] = res
        val = (
            round(res["Average"]["Test Acc"], 4)
            if "Average" in res
            else round(res["Test Acc"], 4)
        )
        bar.set_postfix_str(f"{name}: {val}")

    return results


def run_expriment(graph, GNN_server, m):
    rep = 25

    for partitioning in ["random"]:
        # for partitioning in ["random", "louvain", "kmeans"]:
        if partitioning == "random":
            # num_subgraphs_list = [10]
            num_subgraphs_list = [10]
        else:
            num_subgraphs_list = [10]

        for num_subgraphs in num_subgraphs_list:
            for train_ratio in [0.1, 0.3, 0.5]:
                test_ratio = 0.9 - train_ratio

                simulation_path = (
                    "./results/Simulation_change_r3/"
                    f"{config.dataset.dataset_name}/"
                    f"{partitioning}/"
                    f"{m}/"
                    f"{num_subgraphs}/"
                    f"{train_ratio:0.2f}/"
                )

                os.makedirs(simulation_path, exist_ok=True)

                LOGGER = getLOGGER(
                    name=f"average_{now}_{config.dataset.dataset_name}",
                    log_on_file=True,
                    save_path=simulation_path,
                )
                LOGGER2 = getLOGGER(
                    name=f"all_{now}_{config.dataset.dataset_name}",
                    terminal=False,
                    log_on_file=True,
                    save_path=simulation_path,
                )
                LOGGER.info(json.dumps(config.config, indent=4))

                bar = tqdm(total=rep)
                results = []
                for i in range(rep):
                    create_clients(
                        graph,
                        GNN_server,
                        train_ratio=train_ratio,
                        test_ratio=test_ratio,
                        num_subgraphs=num_subgraphs,
                        partitioning=partitioning,
                    )
                    model_results = {}

                    GNN_result = get_GNN_results2(GNN_server, bar=bar)
                    model_results.update(GNN_result)

                    LOGGER2.info(f"Run id: {i}")
                    LOGGER2.info(json.dumps(model_results, indent=4))

                    results.append(model_results)

                    average_result = calc_average_std_result(results)
                    file_name = (
                        f"{simulation_path}{now}_{config.dataset.dataset_name}.csv"
                    )
                    save_average_result(average_result, file_name)

                    bar.update()

                LOGGER.info(json.dumps(average_result, indent=4))

                LOGGER.handlers.clear()
                LOGGER2.handlers.clear()


if __name__ == "__main__":
    graph = define_graph(config.dataset.dataset_name)

    GNN_server = GNNServer(graph)
    m_range = [40, 200, 180, 100, 80, 60, 160]

    for m in m_range:
        config.spectral.spectral_len = m
        config.spectral.lanczos_iter = m

        run_expriment(graph, GNN_server, m)
