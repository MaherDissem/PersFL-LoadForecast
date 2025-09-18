# Personalized Federated Learning Framework for Load Forecasting

This repository implements a federated learning framework for time series forecasting and incorporates various FL personalization mechanisms to enhance performance for non-IID clients.

## Methods Implemented

- **FedAvg based Federated Learning**: A decentralized approach where multiple clients collaboratively train a model without sharing their raw data.
- **Adaptive Local Aggregation (FedALA)**: Optimizes the initialization of the client model in each FL round.
- **Model-Mix Client Personalization**: Achieves personalization by building for each client a private local model and a public one that participates in FL, and mixing their weights.
- **Client Clustering**: An optional algorithm to divide clients into clusters based on load patterns similarity without sharing their data.
- The implementation of the **Fedora** PFL algorithm for load forecasting is available [here](https://github.com/MaherDissem/FEDORA/tree/load-forecast).

## Datasets

In these experiments, we use the IRISE and REFIT datasets. While IRISE is not publicly available, the REFIT dataset can be obtained [here](https://tokhub.github.io/dbecd/links/redd.html). It should be stored in `data/raw/CLEAN_REFIT_081116`.

## File Overview
```
project_root/ 
├── src/                             # Directory containing the source code.  
│   ├── simulation.py                # Main script to start the FL simulation.  
│   ├── config.py                    # Contains configuration parameters for the simulation's setup.  
│   ├── server.py                    # Defines the server-side strategy for FL, including aggregation of client updates.  
│   ├── communication.py             # Manages communication protocols between clients and the server.  
│   ├── clients/                     # Contains client-related files for FL.  
│   │   ├── client.py                # Entry point for creating client objects and defining interaction with the server.  
│   │   ├── base_model.py            # Defines client logic for standard FL without personalization.  
│   │   ├── ALA.py                   # Implements Adaptive Local Aggregation for initializing the client model in each FL round.  
│   │   └── mixed_model.py           # Implements a forecasting model that combines local and federated models for personalization.  
│   ├── forecasting/                 # Contains different forecasting model implementations.  
│   │   ├── SCINet/                  # Directory for SCINet model implementation.
│   │   │   ├── SCINet.py            # Defines the model.
│   │   │   └── wrapper.py           # Handles the model training and evaluation logic.
│   │   ├── seq2seq/                 # Directory for Seq2Seq model implementation.
│   │   │   ├── model.py             # Defines the model.
│   │   │   └── wrapper.py           # Handles the model training and evaluation logic.
│   │   └── early_stop.py            # Implements early stopping logic to prevent overfitting.  
│   ├── isolated_client.py           # Implements logic for a client operating in isolation without FL.
│   ├── preprocess_dataset.py        # Preprocesses raw data.
│   ├── dataset.py                   # Handles data loading for the clients.  
│   ├── metrics.py                   # Defines custom evaluation metrics for the FL process.  
│   ├── utils.py                     # Utility functions for logging, setting seeds, and preparing folders.  
│   └── notebooks/                   # Contains Jupyter notebooks.
│   │   ├── load_clustering.ipynb    # Visualizes the client clustering process.
│   │   └── results_analysis.ipynb   # Aanalyzes and visualizes the results.
├── .gitignore                       # Specifies files and directories to be ignored by Git.  
├── LICENSE                          # License file for the project.  
├── requirements.txt                 # Lists project dependencies for easy installation.  
└── README.md                        # Overview of the project, including methods implemented and how to run the simulation.  
```

## Running the Simulation

1. **Set up a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Note: this repository was developed with the `1.8.0` version of [Flower](https://flower.ai/) and might not work out of the box for later versions.

3. **Download and preprocess the data:**

    Place the raw dataset files in the `data/raw` directory. Each file represents a different client.
    These files will be preprocessed (normalized, resampled to hourly frequency and NaN values handled) using the following command:
    ```bash
    python src/preprocess_dataset.py
    ```

4. **Configure parameters:** 

    Modify `src/config.py` to set desired parameters. Refer to the comments in that file for a descriptions of each parameter.


5. **Run the simulation:**
    ```bash
    python src/simulation.py
    ```
    Results will be saved to `simulations/<simulation_id>/` and can be analyzed with `src/notebooks/results_analysis.ipynb`.

## Citation
If you find this repository useful, please consider citing it as follows:
```
@ARTICLE{11037499,
  author={Dissem, Maher and Amayri, Manar},
  journal={IEEE Internet of Things Journal}, 
  title={Toward Efficient Federated Load Forecasting: Personalization Mechanisms and Their Impact}, 
  year={2025},
  volume={12},
  number={17},
  pages={36045-36062},
  keywords={Data models;Load forecasting;Adaptation models;Forecasting;Federated learning;Load modeling;Predictive models;Computational modeling;Training;Smart buildings;Load forecasting;personalized federated learning (PFL);smart buildings},
  doi={10.1109/JIOT.2025.3580378}
}
```
