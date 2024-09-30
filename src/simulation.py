import os
import sys
import flwr as fl

from config import config
from server import FedCustom
from clients.client import client_fn
from metrics import evaluate
from utils import fit_config, set_seed
from flwr.common.logger import log
from utils import clean_folder


os.environ["PYTHONPATH"] = os.pathsep.join(
    sys.path
)  # Ensure environment variable is set for subprocesses
set_seed(config.seed)

# Prepare the experiments folder
for folder in [config.weights_folder_path, config.results_folder_path]:
    clean_folder(folder)

# Configure the logger
fl.common.logger.configure(identifier="FlowerExperiment", filename=config.log_file)
log(msg=config(), level=20)

# Start the simulation
fed_sparse_strategy = FedCustom(
    fraction_fit=config.fraction_fit,
    fraction_evaluate=config.fraction_evaluate,
    min_available_clients=config.nbr_clients,
    min_fit_clients=config.min_fit_clients,
    min_evaluate_clients=config.min_evaluate_clients,
    evaluate_fn=evaluate,  # cusom server-side evaluation function
    on_fit_config_fn=fit_config,  # function called before each round of training
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=config.nbr_clients,
    config=fl.server.ServerConfig(num_rounds=config.nbr_rounds),
    strategy=fed_sparse_strategy,
    client_resources={"num_cpus": config.num_cpus, "num_gpus": config.num_gpus},
)
