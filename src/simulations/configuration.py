from src.GNN.GNN_server import GNNServer


def get_configurations(
    GNN_server: GNNServer, normal_epoch, laplace_epoch, spectral_epoch
):
    configurations = {
        "server_GNN": {
            "method": GNN_server.train_local_model,
            "epochs": normal_epoch,
            "data_type": "feature",
            "f_type": "GNN",
            "s_type": None,
            "FL": None,
        },
        "local_GNN": {
            "method": GNN_server.joint_train_g,
            "epochs": normal_epoch,
            "data_type": "feature",
            "f_type": "GNN",
            "s_type": None,
            "FL": False,
        },
        "flga_GNN": {
            "method": GNN_server.joint_train_g,
            "epochs": normal_epoch,
            "data_type": "feature",
            "f_type": "GNN",
            "s_type": None,
            "FL": True,
        },
        # "flga_S_MLP": {"method": GNN_server.joint_train_g, "epochs": normal_epoch, "data_type": "structure", "s_type": "MLP", "FL": True},
        "flga_S_DGCN": {
            "method": GNN_server.joint_train_g,
            "epochs": normal_epoch,
            "data_type": "structure",
            "f_type": None,
            "s_type": "DGCN",
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
        "flga_S_Spectral": {
            "method": GNN_server.joint_train_g,
            "epochs": spectral_epoch,
            "data_type": "structure",
            "f_type": None,
            "s_type": "SpectralLaplace",
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
        "flga_DGCN_F+S_Spectral": {
            "method": GNN_server.joint_train_g,
            "epochs": spectral_epoch,
            "data_type": "f+s",
            "f_type": "DGCN",
            "s_type": "SpectralLaplace",
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
