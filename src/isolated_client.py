import pandas as pd
from torch.utils.data import DataLoader

from forecasting.seq2seq.wrapper import ModelWrapper as Seq2Seq
from forecasting.SCINet.wrapper import ModelWrapper as SCINet
from config import config
from utils import set_seed
from dataset import get_experiment_data


def run_on_local_data(
    trainloader: DataLoader, validloader: DataLoader, testloader: DataLoader
):
    """Train and test a forecasting model on local data only."""

    if config.model == "SCINet":
        local_model_wrapper = SCINet(config, config.input_size, config.forecast_horizon)
    elif config.model == "Seq2Seq":
        local_model_wrapper = Seq2Seq(
            config, config.input_size, config.forecast_horizon
        )
    else:
        raise NotImplementedError("Model not implemented")

    loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = (
        local_model_wrapper.train(trainloader, validloader, testloader)
    )
    return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


def eval_isolated_client():
    """Evaluate a model not participating in federated learning.
    This is used for comparing performance of local training vs federated learning."""

    set_seed(config.seed)
    trainloaders, valloaders, testloaders, dataset_paths, min_vals, max_vals = (
        get_experiment_data(
            data_root=config.data_root,
            num_clients=config.nbr_clients,
            input_size=config.input_size,
            forecast_horizon=config.forecast_horizon,
            stride=config.stride,
            batch_size=config.batch_size,
            valid_set_size=config.valid_set_size,
            test_set_size=config.test_set_size,
        )
    )
    results = pd.DataFrame(columns=["cid", "smape", "mae", "mse", "rmse", "r2"])
    for cid in range(config.nbr_clients):
        loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = (
            run_on_local_data(trainloaders[cid], valloaders[cid], testloaders[cid])
        )
        new_row = pd.DataFrame(
            {
                "cid": [cid],
                "dataset_path": [dataset_paths[cid]],
                "smape": [smape_loss],
                "mae": [mae_loss],
                "mse": [mse_loss],
                "rmse": [rmse_loss],
                "r2": [r2_loss],
            }
        )
        results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv("local_results.csv", index=False)


if __name__ == "__main__":
    eval_isolated_client()
