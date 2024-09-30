from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import INFO, WARNING
import os
import pandas as pd
import torch
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)

from config import config
from clients.base_model import ForecastingModel
from metrics import evaluate
from communication import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
from utils import set_seed
from utils import plot_cluster_centroids

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = config.fraction_fit,
        fraction_evaluate: float = config.fraction_evaluate,
        min_available_clients: int = config.min_available_clients,
        min_fit_clients: int = config.min_fit_clients,
        min_evaluate_clients: int = config.min_evaluate_clients,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = evaluate,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Custom FedAvg strategy with sparse matrices based communication.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__()
        set_seed(config.seed)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return "FedCustom"

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters (server)."""
        model = ForecastingModel(
            config=config,
            trainloader=None,
            validloader=None,
            testloader=None,
        )
        ndarrays = model.get_parameters()
        return fl.common.ndarrays_to_parameters(ndarrays)

    def init_cluster_centroids(self):
        set_seed(88)
        self.cluster_centroids = []
        for k in range(config.n_clusters):
            self.cluster_centroids.append(torch.rand(config.clustering_seq_len, 1))
        return self.cluster_centroids

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = (
            client_manager.num_available()
            if server_round <= config.nbr_clustering_rounds
            else sample_size
        )  # Call all clients for clustering
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Set up the configuration for each client
        if config.cluster_clients:
            # Clustering rounds
            if server_round <= config.nbr_clustering_rounds:
                log(INFO, f"Server round {server_round}: Clustering round")
                fit_configurations = []
                ins_config = {
                    "server_round": server_round,
                }
                parameters = (
                    self.init_cluster_centroids()
                    if server_round == 1
                    else self.cluster_centroids
                )
                parameters = ndarrays_to_sparse_parameters(parameters)
                for client in clients:
                    fit_configurations.append((client, FitIns(parameters, ins_config)))
                return fit_configurations

            # Training rounds
            # inter-cluster rounds
            if (
                server_round > config.nbr_clustering_rounds
                and server_round
                <= config.nbr_clustering_rounds + config.nbr_inter_cluster_rounds
            ):
                log(INFO, f"Server round {server_round}: Clustered training round")
                fit_configurations = []
                for client in clients:
                    # Send cluster parameters to corresponding clients
                    parameters = self.cluster_weights[
                        self.cluster_assignments[client.cid]
                    ]
                    ins_config = {"server_round": server_round}
                    fit_configurations.append((client, FitIns(parameters, ins_config)))
                return fit_configurations

        # Global model training rounds after clustering or clustering is disabled
        else:
            log(INFO, f"Server round {server_round}: Global training round")
            fit_configurations = []
            for client in clients:
                # Send global model parameters to all clients
                ins_config = {"server_round": server_round}
                fit_configurations.append((client, FitIns(parameters, ins_config)))
            return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average (FedAvg algorithm)."""
        if not results:
            log(WARNING, f"Server round {server_round}: No results to aggregate")
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if config.cluster_clients:
            # Clustering rounds
            if server_round < config.nbr_clustering_rounds:
                # Aggregate clinet submissions
                j_clients = []
                v_clients = []
                for _, fit_res in results:
                    # Undo the sparse matrix transformation used to save bandwidth
                    stacked_j_v = sparse_parameters_to_ndarrays(fit_res.parameters)
                    stacked_j_v = torch.stack(
                        [
                            (
                                torch.tensor(arr)
                                if arr.shape[-1] > 0
                                else torch.zeros(config.clustering_seq_len + 1, 1)
                            )
                            for arr in stacked_j_v
                        ]
                    )
                    j = stacked_j_v[:, :-1, :]
                    v = stacked_j_v[:, -1, :].squeeze()
                    j_clients.append(j)
                    v_clients.append(v)
                j_sum = torch.sum(torch.stack(j_clients), dim=0)
                v_sum = torch.sum(torch.stack(v_clients), dim=0)
                # Update cluster centroids
                for k in range(config.n_clusters):
                    self.cluster_centroids[k] += config.clustering_alpha * (
                        j_sum[k] / v_sum[k] if v_sum[k] else 0
                    )
                return None, {}

            # Cluster assignement round
            if server_round == config.nbr_clustering_rounds:
                log(INFO, "Clustering done.")
                # Find what cluster each client assigned themselves to
                self.cluster_assignments = {}
                for _, fit_res in results:
                    cid, cluster = fit_res.metrics["cid"], fit_res.metrics["cluster_id"]
                    self.cluster_assignments[cid] = cluster
                # Plot cluster centroids
                plot_cluster_centroids(
                    self.cluster_centroids,
                    config.n_clusters,
                    path=os.path.join(config.sim_name, "cluster_centroids.png"),
                )
                # Initialize cluster weights
                self.cluster_weights = {}
                for cluster_id in range(config.n_clusters):
                    self.cluster_weights[cluster_id] = self.initialize_parameters(None)
                return None, {}

            # Training rounds: inter-cluster weights aggregation
            # We deserialize the results with our custom method
            weights_results = [
                (
                    sparse_parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                    fit_res.metrics["cid"],
                )
                for _, fit_res in results
            ]

            # Group by cluster_id
            cluster_results = {}
            for weights, num_examples, cid in weights_results:
                cluster_id = self.cluster_assignments[cid]
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = []
                cluster_results[cluster_id].append((weights, num_examples))

            # Aggregate client models by cluster
            self.cluster_weights = {}
            cluster_weights_n_samples = {}
            for cluster_id, cluster_results in cluster_results.items():
                aggr_weights = aggregate(cluster_results)
                serialized_weights = ndarrays_to_sparse_parameters(aggr_weights)
                self.cluster_weights[cluster_id] = serialized_weights

                cluster_total_samples = sum(
                    [num_examples for _, num_examples in cluster_results]
                )
                cluster_weights_n_samples[cluster_id] = (
                    aggr_weights,
                    cluster_total_samples,
                )

            # Aggregate cluster weights into global model: intra-cluster aggregation
            parameters_aggregated = aggregate(cluster_weights_n_samples.values())

        else:
            # Client aggregation is diabled
            # We deserialize the results with our custom method
            weights_results = [
                (
                    sparse_parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                )
                for _, fit_res in results
            ]

            # We aggregate and serialize results back
            parameters_aggregated = aggregate(weights_results)

        # Serialize the aggregated weights
        serialized_aggr_parameters = ndarrays_to_sparse_parameters(
            parameters_aggregated
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return serialized_aggr_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation. Provide new model parameters and send instructions via config."""
        if self.fraction_evaluate == 0.0:
            return []
        config_ins = {
            "server_round": server_round,
        }
        evaluate_ins = EvaluateIns(parameters, config_ins)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        if server_round == config.nbr_rounds:  # For the last round, sample all clients
            sample_size = client_manager.num_available()
            min_num_clients = client_manager.num_available()

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate metrics
        metrics_aggregated = {}
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            cid = evaluate_res.metrics["cid"]
            metrics.pop("cid")
            metrics_aggregated[cid] = metrics

        # Save metrics to a csv file
        results = pd.DataFrame(
            columns=["round", "eval_data", "cid", "smape", "mae", "mse", "rmse", "r2"]
        )
        for cid, metrics in metrics_aggregated.items():
            test_row = pd.DataFrame(
                {
                    "round": [server_round],
                    "eval_data": ["test"],
                    "cid": [int(cid)],
                    "smape": [metrics["test_smape"]],
                    "mae": [metrics["test_mae"]],
                    "mse": [metrics["test_mse"]],
                    "rmse": [metrics["test_rmse"]],
                    "r2": [metrics["test_r2"]],
                }
            )
            val_row = pd.DataFrame(
                {
                    "round": [server_round],
                    "eval_data": ["val"],
                    "cid": [int(cid)],
                    "smape": [metrics["val_smape"]],
                    "mae": [metrics["val_mae"]],
                    "mse": [metrics["val_mse"]],
                    "rmse": [metrics["val_rmse"]],
                    "r2": [metrics["val_r2"]],
                }
            )
            results = pd.concat([results, test_row, val_row], ignore_index=True)
        results.sort_values(by=["round", "eval_data", "cid"], inplace=True)
        os.makedirs(config.results_folder_path, exist_ok=True)
        results.to_csv(
            f"{config.results_folder_path}/results.csv",
            index=False,
            mode="a",
            header=server_round == 1,
        )

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate parameters on server-side model using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # We deserialize using our custom method
        parameters_ndarrays = sparse_parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics
