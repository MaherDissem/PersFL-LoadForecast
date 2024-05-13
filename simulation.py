import flwr as fl
from client import client_fn
from config import NUM_CLIENTS, client_resources
from server import strategy


# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)