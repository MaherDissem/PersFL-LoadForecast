import flwr as fl

from config import config
from client import client_fn
from metrics import evaluate
from utils import fit_config
from server import FedCustom


# from server import fedavg_strategy
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=nbr_clients,
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=fedavg_strategy,
#     client_resources=client_resources,
# )

fed_sparse_strategy = FedCustom(
    fraction_fit=config.fraction_fit,
    fraction_evaluate=config.fraction_evaluate,
    min_available_clients=config.nbr_clients,
    min_fit_clients=config.min_fit_clients,
    min_evaluate_clients=config.min_evaluate_clients,
    evaluate_fn=evaluate,  # cusom server-side evaluation function
    on_fit_config_fn=fit_config,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=config.nbr_clients,
    config=fl.server.ServerConfig(num_rounds=config.nbr_rounds),
    strategy=fed_sparse_strategy,  # <-- pass the new strategy here
    client_resources=config.client_resources,
)
