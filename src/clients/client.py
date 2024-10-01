from typing import List, Tuple
from logging import ERROR
import os
import numpy as np
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)
from flwr.common.logger import log
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import config
from dataset import get_experiment_data
from communication import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
from clients.base_model import ForecastingModel
from clients.mixed_model import PersForecastingModel
from utils import set_seed


class FlowerClient(fl.client.Client):
    def __init__(
        self,
        config,
        cid,
        model,
        trainloader,
        validloader,
        testloader,
        dataset_path,
        min_val,
        max_val,
    ):
        set_seed(config.seed)
        self.config = config
        self.cid = cid
        self.model = model
        # Dataloaders
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.dataset_path = dataset_path
        self.min_val = min_val
        self.max_val = max_val
        # clustering
        self.cluster = None
        self.n_clusters = self.config.n_clusters
        self.seq_len = self.config.input_size + self.config.forecast_horizon

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current parameters of the client's federated model."""
        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = self.model.get_parameters()

        # Serialize ndarray's into a Parameters object using our custom function
        parameters = ndarrays_to_sparse_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Set the parameters of the client's federated model to the server parameters, train the model, and return the updated parameters of the federated component."""
        ins_config = ins.config
        server_round = ins_config["server_round"]
        parameters_original = ins.parameters
        # Deserialize parameters to NumPy ndarray's
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

        if self.config.cluster_clients:
            # Clustering round
            if server_round < self.config.nbr_clustering_rounds:
                cluster_centroids = ndarrays_original
                Ploader = self._get_client_p_matrix()
                j, v = self._client_subroutine(
                    Ploader, cluster_centroids
                )  # j.shape: torch.Size([3, 168, 1]), v.shape: torch.Size([3])
                stacked_j_v = torch.cat(
                    (j, v.unsqueeze(1).unsqueeze(2)), dim=1
                )  # Shape: [3, 168 + 1, 1]
                return FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=ndarrays_to_sparse_parameters(stacked_j_v),
                    num_examples=self.model.len_trainloader,
                    metrics={},
                )

            # Centroids found: assign cluster
            # No training is done in this round because all clients need first to receive the weights of the cluster they chose
            if server_round == self.config.nbr_clustering_rounds:
                cluster_centroids = ndarrays_original
                cluster_id = self._assign_cluster(cluster_centroids)
                self.cluster = cluster_id.item()
                if config.filter_outliers:
                    Ploader = self._get_client_p_matrix()
                    filtered_train_dataset = self._update_model_trainloader(
                        Ploader, cluster_centroids, config.outliers_threshold
                    )
                    torch.save(  # Data is kept private and not shared with the server
                        filtered_train_dataset,
                        f"{config.filtered_data_path}/{self.cid}.pt",
                    )
                return FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=ndarrays_to_sparse_parameters(
                        torch.tensor((0, 0))
                    ),  # Send empty Parameters object
                    num_examples=self.model.len_trainloader,
                    metrics={
                        "cid": self.cid,
                        "cluster_id": self.cluster,
                    },  # These are not metric but this is the only way to pass them to the server
                )

        # Training round: set local model, train and get updated parameters
        if config.cluster_clients and config.filter_outliers:
            self._load_trainloader()
        self.model.set_parameters(ndarrays_original)
        self.model.train()
        ndarrays_updated = self.model.get_parameters()

        # Build and return response
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_sparse_parameters(
                ndarrays_updated
            ),  # Serialize ndarray's into a Parameters object using our custom function
            num_examples=self.model.len_trainloader,
            metrics={"cid": self.cid},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Set the parameters of the client's federated model to the server parameters, retrain and evaluate the model, and return the evaluation metrics."""
        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

        # Load server model into local client model
        self.model.set_parameters(ndarrays_original)

        # Evaluate local model:
        # train the local model by mixing it with the newly aggregated parameters. Metrics returned are for the test data.
        if (
            self.config.cluster_clients
            and ins.config["server_round"] <= self.config.nbr_clustering_rounds
        ):
            test_smape, test_mae, test_mse, test_rmse, test_r2 = 0, 0, 0, 0, 0
            val_smape, val_mae, val_mse, val_rmse, val_r2 = 0, 0, 0, 0, 0
        else:
            if config.cluster_clients and config.filter_outliers:
                self._load_trainloader()
            _, test_smape, test_mae, test_mse, test_rmse, test_r2 = self.model.train()
            val_smape, val_mae, val_mse, val_rmse, val_r2 = self.model.evaluate()
        loss = test_smape
        metrics = {
            "cid": self.cid,  # not a metric, but useful for evaluation
            "test_smape": test_smape,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "val_smape": val_smape,
            "val_mae": val_mae,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=self.model.len_validloader,
            metrics=metrics,
        )

    def _get_client_p_matrix(self) -> DataLoader:
        p_list = []
        for x, y in self.trainloader:
            z = torch.cat((x, y), dim=1)
            p_list.extend(z)
        Pdataset = TensorDataset(torch.stack(p_list))
        Ploader = DataLoader(
            Pdataset, batch_size=1
        )  # batch_size is hardcoded to 1, must not be changed
        return Ploader

    def _client_subroutine(
        self,
        Ploader: DataLoader,
        cluster_centroids: List[torch.tensor],
    ) -> Tuple[torch.tensor, torch.tensor]:
        closest_centroid_id_list = []
        closest_centroid_distances = []
        closest_centroid_euclidean_dist = []

        for p_d in Ploader:
            distances = []
            for k in range(self.n_clusters):
                centroid_k = cluster_centroids[k]
                dist = torch.norm(p_d[0] - centroid_k)
                distances.append(dist)
            min_dist, min_idx = torch.min(torch.tensor(distances), dim=0)
            closest_centoid = cluster_centroids[min_idx]
            diff = p_d[0].squeeze(0) - closest_centoid

            closest_centroid_id_list.append(min_idx)
            closest_centroid_distances.append(diff)
            closest_centroid_euclidean_dist.append(min_dist)

        j_list = [torch.zeros(self.seq_len, 1) for _ in range(self.n_clusters)]
        v_list = [0 for _ in range(self.n_clusters)]
        for centoid_id, centroid_distance in zip(
            closest_centroid_id_list, closest_centroid_distances
        ):
            j_list[centoid_id] += centroid_distance
            v_list[centoid_id] += 1
        return torch.stack(j_list, dim=0), torch.tensor(v_list)

    def _filter_outliers(
        self,
        Ploader: DataLoader,
        cluster_centroids: List[torch.tensor],
        distance_percentile: float,
    ) -> DataLoader:
        distances = []
        for p_d in Ploader:
            sample_distances = []
            for k in range(self.n_clusters):
                centroid_k = cluster_centroids[k]
                dist = torch.norm(p_d[0] - centroid_k)
                sample_distances.append(dist)
            min_dist, min_idx = torch.min(torch.tensor(sample_distances), dim=0)
            distances.append(min_dist)

        threshold = torch.quantile(torch.tensor(distances), distance_percentile)
        filtered_data = []
        for p_d, dist in zip(Ploader, distances):
            if dist < threshold:
                filtered_data.append(p_d[0].squeeze(0))

        filtered_dataset = TensorDataset(torch.stack(filtered_data))
        filtered_loader = DataLoader(filtered_dataset, batch_size=1)
        return filtered_loader

    def _compute_avg_load(self, Ploader: DataLoader) -> torch.tensor:
        n_samples = 0
        total_load = torch.zeros(self.seq_len, 1)
        for p_d in Ploader:
            total_load += p_d[0].squeeze(0)
            n_samples += 1
        return total_load / n_samples

    def _assign_cluster(self, cluster_centroids: List[torch.tensor]) -> int:
        Ploader = self._get_client_p_matrix()
        if self.config.filter_outliers:
            filtered_Ploader = self._filter_outliers(
                Ploader, cluster_centroids, self.config.outliers_threshold
            )
            Ploader = filtered_Ploader

        avg_load = self._compute_avg_load(Ploader)
        distances = []
        for k in range(self.n_clusters):
            centroid_k = cluster_centroids[k]
            dist = torch.norm(avg_load - centroid_k)
            distances.append(dist)
        min_dist, min_idx = torch.min(torch.tensor(distances), dim=0)
        self.cluster = min_idx
        return min_idx

    def _update_model_trainloader(
        self,
        Ploader: DataLoader,
        cluster_centroids: List[torch.tensor],
        distance_percentile: float,
    ) -> DataLoader:
        """Replace the trainloader with a new one that filters out outlying sequences.
        Valid and test loaders are not affected.
        """
        filtered_Ploader = self._filter_outliers(
            Ploader,
            cluster_centroids,
            # self.config.outliers_threshold,
            distance_percentile,
        )
        x_data = []
        y_data = []
        for filter_batch in filtered_Ploader:
            x = filter_batch[0].squeeze(0)[: -self.config.forecast_horizon, :]
            y = filter_batch[0].squeeze(0)[-self.config.forecast_horizon :, :]
            x_data.append(x)
            y_data.append(y)
        x_data = torch.stack(x_data)
        y_data = torch.stack(y_data)

        filtered_train_dataset = TensorDataset(x_data, y_data)
        return filtered_train_dataset

    def _load_trainloader(self) -> None:
        try:
            filtered_train_dataset = torch.load(
                f"{config.filtered_data_path}/{self.cid}.pt"
            )
            self.trainloader = DataLoader(
                filtered_train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
            self.model.trainloader = self.trainloader
        except Exception as e:
            log(ERROR, f"Error loading filtered data: {e}")


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client from a single building's data (identified by cid)."""
    # Load data
    trainloaders, valloaders, testloaders, csv_paths, min_vals, max_vals = (
        get_experiment_data(
            data_root=config.data_root,
            num_clients=config.nbr_clients,
            input_size=config.input_size,
            forecast_horizon=config.forecast_horizon,
            stride=config.stride,
            batch_size=config.batch_size,
            valid_set_size=config.valid_set_size,
            test_set_size=config.test_set_size,
        )
    )
    trainloader = trainloaders[int(cid)]
    validloader = valloaders[int(cid)]
    testloader = testloaders[int(cid)]
    csv_path = csv_paths[int(cid)]
    min_val = min_vals[int(cid)]
    max_val = max_vals[int(cid)]
    # Load model
    os.makedirs(config.weights_folder_path, exist_ok=True)
    config.checkpoint_path = os.path.join(
        config.weights_folder_path, f"model_{cid}.pth"
    )
    config.seed = config.seed + int(cid)
    if config.personalization:
        model = PersForecastingModel(
            config,
            trainloader,
            validloader,
            testloader,
        )
    else:
        model = ForecastingModel(
            config,
            trainloader,
            validloader,
            testloader,
        )
    # Create a Flower client representing representing a single building (single data source)
    return FlowerClient(
        config,
        cid,
        model,
        trainloader,
        validloader,
        testloader,
        csv_path,
        min_val,
        max_val,
    )
