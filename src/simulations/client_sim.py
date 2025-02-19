import os
import sys
import json


pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

from tqdm import tqdm

from src import *
from src.FedGCN.FedGCN_server import FedGCNServer
from src.FedPub.fedpub_server import FedPubServer

# from src.simulations.configuration import get_configurations
from src.simulations.simulation_utils import (
    calc_average_std_result,
    get_Fedgcn_results,
    get_Fedpub_results,
    save_average_result,
)
from src.utils.define_graph import define_graph
from src.utils.logger import getLOGGER
from src.GNN.GNN_server import GNNServer
from src.utils.graph_partitioning import fedGCN_partitioning, partition_graph


def get_configurations(
    GNN_server: GNNServer, normal_epoch, laplace_epoch, spectral_epoch
):
    configurations = {
        "flga_GNN": {
            "method": GNN_server.joint_train_g,
            "epochs": normal_epoch,
            "data_type": "feature",
            "f_type": "GNN",
            "s_type": None,
            "FL": True,
        },
        "flga_S_Laplacian": {
            "method": GNN_server.joint_train_g,
            "epochs": laplace_epoch,
            "data_type": "structure",
            "f_type": None,
            "s_type": "Laplace",
            "FL": True,
        },
        "flga_S_Arnoldi": {
            "method": GNN_server.joint_train_g,
            "epochs": spectral_epoch,
            "data_type": "structure",
            "f_type": None,
            "s_type": "LanczosLaplace",
            "FL": True,
        },
        # "flga_DGCN_F+S_MLP": {"method": GNN_server.joint_train_g, "epochs": normal_epoch, "data_type": "structure", "f_type": "DGCN", "s_type": "MLP", "FL": True},
        "flga_DGCN_F+S_DGCN": {
            "method": GNN_server.joint_train_g,
            "epochs": normal_epoch,
            "data_type": "f+s",
            "f_type": "DGCN",
            "s_type": "DGCN",
            "FL": True,
        },
        "flga_DGCN_F+S_Laplacian": {
            "method": GNN_server.joint_train_g,
            "epochs": laplace_epoch,
            "data_type": "f+s",
            "f_type": "DGCN",
            "s_type": "Laplace",
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
    graph,
    GNN_server: GNNServer,
    FedPub_server: FedPubServer,
    Fedgcn_server2: FedGCNServer,
    train_ratio,
    test_ratio,
    num_subgraphs,
    partitioning,
):
    graph.add_masks(train_ratio=train_ratio, test_ratio=test_ratio)

    GNN_server.reset_clients()

    FedPub_server.reset_clients()
    Fedgcn_server2.reset_clients()

    subgraphs = partition_graph(graph, num_subgraphs, partitioning)

    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)
        FedPub_server.add_client(subgraph)

    fedgcn_subgraphs2 = fedGCN_partitioning(
        graph, config.subgraph.num_subgraphs, method=partitioning, num_hops=2
    )
    for subgraph in fedgcn_subgraphs2:
        Fedgcn_server2.add_client(subgraph)


def get_GNN_results2(
    GNN_server: GNNServer,
    bar: tqdm,
    normal_epoch=config.model.iterations,
    laplace_epoch=100,
    spectral_epoch=config.model.iterations,
):
    results = {}

    # res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    # results[f"FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    # results[f"FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    # results[f"FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    configurations = get_configurations(
        GNN_server, normal_epoch, laplace_epoch, spectral_epoch
    )
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


def run_expriment(graph, GNN_server, FedPub_server, FedGCN_server2):
    rep = 10
    epochs = config.model.iterations
    fedpub_epochs = config.fedpub.epochs

    for partitioning in ["random"]:
        # for partitioning in ["random", "louvain", "kmeans"]:

        num_subgraphs_list = np.arange(5, 55, 5)

        for num_subgraphs in num_subgraphs_list:
            for train_ratio in [config.subgraph.train_ratio]:
                test_ratio = config.subgraph.test_ratio

                simulation_path = (
                    "./results/Simulation_clients/"
                    f"{config.dataset.dataset_name}/"
                    f"{partitioning}/"
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
                        FedPub_server,
                        FedGCN_server2,
                        train_ratio=train_ratio,
                        test_ratio=test_ratio,
                        num_subgraphs=num_subgraphs,
                        partitioning=partitioning,
                    )
                    model_results = {}

                    Fedpub_results = get_Fedpub_results(
                        FedPub_server,
                        bar=bar,
                        epochs=fedpub_epochs,
                    )
                    model_results.update(Fedpub_results)

                    Fedgcn_results2 = get_Fedgcn_results(
                        FedGCN_server2,
                        bar=bar,
                        epochs=epochs,
                        num_hops=2,
                    )
                    model_results.update(Fedgcn_results2)

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
    FedPub_server = FedPubServer(graph)

    FedGCN_server2 = FedGCNServer(graph)

    run_expriment(graph, GNN_server, FedPub_server, FedGCN_server2)
