from dataclasses import dataclass
import torch

@dataclass
class config: # TODO upper case
    """Configuration class for the federated learning setup."""

    # Clients parameters
    data_root: str = "data/processed"
    nbr_clients: int = 20
    personalization: bool = True

    # Dataloader parameters
    batch_size: int = 32
    valid_set_size: int = 0.15
    test_set_size: int = 0.15

    # Server parameters
    nbr_rounds: int = 30 # nbr_clustering_rounds + nbr_inter_cluster_rounds + nbr_global_rounds
    fraction_fit: float = 20 / 20       # TODO make lower
    fraction_evaluate: float = 20 / 20  # TODO make lower
    min_available_clients: int = 3
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2

    # Experiment results
    log_file: str = "log.txt"

    # Resources
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    num_cpus: int = 1
    num_gpus: float = 1.0 if device.type == "cuda" else 0.0

    # Client clustering parameters
    cluster_clients: bool = True
    n_clusters: int = 3
    clustering_seq_len = 24 * 7
    nbr_clustering_rounds: int = 10
    nbr_inter_cluster_rounds: int = 15
    clustering_alpha: float = 1e-0
    filter_outliers: bool = True
    outliers_threshold: float = 0.95

    # Model mixing parameters
    mu: float = 0.01
    nu: float = 2.0
    eval_local: bool = False # after training the mixed model, whether to eval the mix or the local model

    # Forecasting parameters
    model: str = "SCINet"  # "Seq2Seq" or "SCINet"
    input_size: int = 24 * 6
    forecast_horizon: int = 24
    nbr_var: int = 1
    stride: int = 24
    # Forecasting training parameters
    epochs: int = 1
    patience: int = 20
    lr: float = 1e-3
    eval_every: int = 10
    checkpoint_path: str = "weights/model.pth"
    seed: int = 0
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

    def upadate_from_args(self, args):
        """Update the default parameters with the given arguments (e.g received as CLI args)."""
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
