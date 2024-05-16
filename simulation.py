import flwr as fl

from config import NUM_CLIENTS, client_resources
from client import client_fn
from metrics import evaluate
from utils import fit_config
from server import FedCustom


# from server import fedavg_strategy
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=fedavg_strategy,
#     client_resources=client_resources,
# )

fed_sparse_strategy = FedCustom(
    fraction_fit=0.005,
    fraction_evaluate=0.01,
    min_fit_clients=20,
    min_evaluate_clients=40,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate,         # cusom server-side evaluation function
    on_fit_config_fn=fit_config,  # Pass the fit_config function
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fed_sparse_strategy,  # <-- pass the new strategy here
    client_resources=client_resources,
)
