- `client.py` is the entry point that creates client objects and define how they interact with the server.

    It also implements an optional algorithm to divide users into clusters based on load patterns similarity without sharing their data. This is our implementation of [this paper](https://ieeexplore.ieee.org/document/10122655) (see src/notebooks/load_clustering.ipynb).

- `base_model.py` defines the client logic for personalization-free Federated Learning (e.g. FedAvg).

- `mixed_model.py` defines a forecasting model that mixes the weights of a local private model with a federated one to achieve personalization. This is our own implementation of the [SuPerFed paper](https://arxiv.org/abs/2109.07628).

- More to be added.
