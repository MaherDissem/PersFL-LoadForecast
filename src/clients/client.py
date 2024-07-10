from typing import List
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

from config import config
from dataset import get_experiment_data
from communication import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
from clients.base_model import ForecastingModel
from clients.mixed_model import PersForecastingModel
from utils import set_seed


class FlowerClient(fl.client.Client):
    def __init__(self, model, cid, dataset_path, min_val, max_val):
        self.model = model
        self.cid = cid
        self.dataset_path = dataset_path
        self.min_val = min_val
        self.max_val = max_val
        set_seed(config.seed)

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
        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        self.model.set_parameters(ndarrays_original)
        self.model.train()
        ndarrays_updated = self.model.get_parameters()

        # Serialize ndarray's into a Parameters object using our custom function
        parameters_updated = ndarrays_to_sparse_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.model.len_trainloader,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Set the parameters of the client's federated model to the server parameters, retrain and evaluate the model, and return the evaluation metrics."""
        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

        # Load server model into local client model
        self.model.set_parameters(ndarrays_original)

        # Evaluate local model:
        # train the local model trained with new federated parameters. Metrics returned are for the test data.
        _, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self.model.train()
        loss = smape_loss  # TODO FIXME
        metrics = {
            "cid": self.cid,  # not a metric, but useful for evaluation
            "smape": loss,
            "mae": mae_loss,
            "mse": mse_loss,
            "rmse": rmse_loss,
            "r2": r2_loss,
        }

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=self.model.len_validloader,
            metrics=metrics,
        )


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
    # Load model
    os.makedirs("weights", exist_ok=True)
    config.checkpoint_path = f"weights/model_{cid}.pth"
    config.seed = config.seed + int(cid)
    if config.personalization:
        model = PersForecastingModel(
            config=config,
            trainloader=trainloaders[int(cid)],
            validloader=valloaders[int(cid)],
            testloader=testloaders[int(cid)],
        )
    else:
        model = ForecastingModel(
            config=config,
            trainloader=trainloaders[int(cid)],
            validloader=valloaders[int(cid)],
            testloader=testloaders[int(cid)],
        )
    # Create a single Flower client representing representing a single building (single data source)
    return FlowerClient(
        model, cid, csv_paths[int(cid)], min_vals[int(cid)], max_vals[int(cid)]
    )
