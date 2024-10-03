import os
from dataclasses import dataclass
import torch


@dataclass
class config:  # TODO upper case
    """Configuration class for the federated learning setup."""

    # Clients parameters
    data_root: str = "data/processed/Combined"
    nbr_clients: int = 43
    personalization: bool = True
    cluster_clients: bool = True
    seed: int = 0

    # Dataloader parameters
    batch_size: int = 32
    valid_set_size: int = 0.15
    test_set_size: int = 0.15
    input_size: int = 24 * 6
    forecast_horizon: int = 24
    nbr_var: int = 1
    stride: int = 24

    # Server parameters
    nbr_rounds: int = (
        40  # nbr_clustering_rounds + nbr_inter_cluster_rounds + nbr_global_rounds
    )
    fraction_fit: float = 20 / 20
    fraction_evaluate: float = 20 / 20
    min_available_clients: int = 3
    min_fit_clients: int = 1
    min_evaluate_clients: int = 1

    # Client clustering parameters
    n_clusters: int = 4
    nbr_clustering_rounds: int = 10
    nbr_inter_cluster_rounds: int = 30
    clustering_seq_len = 24 * 7
    clustering_alpha: float = 1e-0
    filter_outliers: bool = False
    outliers_threshold: float = 0.90

    # Model mixing parameters
    mu: float = 0.01
    nu: float = 2.0
    eval_local: bool = False  # wether to eval the mixed or the local model

    # Forecasting parameters
    model: str = "SCINet"  # "Seq2Seq" or "SCINet"
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    eval_every: int = 10
    verbose: bool = True
    # Seq2seq2 model parameters (relevant only if model_choice="seq2seq")
    s2s_hidden_size: int = 128
    s2s_num_grulstm_layers: int = 1
    s2s_fc_units: int = 16
    # SCINet model parameters (relevant only if model_choice="scinet")
    L1Loss: bool = True
    lradj: int = 2
    concat_len: int = 0
    hidden_size: float = 1.0
    kernel: int = 5
    positionalEcoding: bool = False
    dropout: float = 0.25
    groups: int = 1
    levels: int = 3
    num_decoder_layer: int = 1
    stacks: int = 2
    long_term_forecast: bool = False
    RIN: bool = False
    decompose: bool = False
    single_step: int = 0
    single_step_output_One: int = 0

    # Experiment results
    sim_name: str = (
        f"simulations/{data_root.split('/')[-1]}_{nbr_clients}_{nbr_rounds}_{model}_{epochs}_{patience}_{'pers' if personalization else 'nopers'}_{'clust' if cluster_clients else 'noclust'}_{n_clusters}"
    )
    log_file: str = os.path.join(sim_name, "log.txt")
    weights_folder_path: str = os.path.join(sim_name, "weights/")
    results_folder_path: str = os.path.join(sim_name, "")
    filtered_data_path: str = os.path.join(sim_name, "filtered_data/")

    # Resources
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    num_cpus: int = 4  # 32 total cores, 4 cores per client, 8 parallel processes
    num_gpus: float = (
        0.5 if device.type == "cuda" else 0.0
    )  # 4 GPUs, 0.5 GPU per client, 8 parallel processes
