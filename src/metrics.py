from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics

from config import config as exp_config
from dataset import get_experiment_data
from model import ForecastingModel


trainloaders, valloaders, testloaders, _, _, _ = get_experiment_data(
    data_root=exp_config.data_root,
    num_clients=exp_config.nbr_clients,
    input_size=exp_config.input_size,
    forecast_horizon=exp_config.forecast_horizon,
    stride=exp_config.stride,
    batch_size=exp_config.batch_size,
    valid_set_size=exp_config.valid_set_size,
    test_set_size=exp_config.test_set_size,
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """This function is called by the server after every round
    to evaluate the current global model on the server-side."""
    sid = 0  # server dataloader index
    model = ForecastingModel(
        config=exp_config,
        trainloader=trainloaders[int(sid)],
        validloader=valloaders[int(sid)],
        testloader=testloaders[int(sid)],
    )
    model.set_parameters(parameters)  # Update model with the latest parameters
    smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = (
        model.test()
    )  # Evaluate the federated model
    loss = smape_loss  # TODO FIXME
    metrics = {
        "smape": loss,
        "mae": mae_loss,
        "mse": mse_loss,
        "rmse": rmse_loss,
        "r2": r2_loss,
    }
    return loss, metrics
