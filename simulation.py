import flwr as fl
from client import client_fn
from config import NUM_CLIENTS, client_resources
from server import fedavg_strategy
from server import FedCustom


# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=fedavg_strategy,
#     client_resources=client_resources,
# )

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=FedCustom(),  # <-- pass the new strategy here
    client_resources=client_resources,
)