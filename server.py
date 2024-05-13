import flwr as fl

from metrics import weighted_average
from client import Net, get_parameters
from config import NUM_CLIENTS
from metrics import evaluate
from utils import fit_config


initial_params = get_parameters(Net())

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.005,
    fraction_evaluate=0.01,
    min_fit_clients=20,
    min_evaluate_clients=40,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=evaluate,         # cusom server-side evaluation function
    on_fit_config_fn=fit_config,  # Pass the fit_config function
)
