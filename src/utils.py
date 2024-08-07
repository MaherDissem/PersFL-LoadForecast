import os
import random
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    # this code is here for referece, these parameters are not used by clients for now
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_cluster_centroids(cluster_centroids: List[torch.tensor], n_clusters: int):
    plt.figure(figsize=(15, 5))
    for k in range(n_clusters):
        plt.plot(cluster_centroids[k].detach().numpy(), label=f"Cluster {k}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Normalized Load")
    plt.title("Centroids of Clusters")
    # plt.show()
    plt.savefig("cluster_centroids.png")
