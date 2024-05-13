import torch


# TODO class config
NUM_CLIENTS = 2
BATCH_SIZE = 32
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1.0, "num_gpus": 0.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.
