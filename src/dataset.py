from typing import List, Tuple
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config


class DatasetForecasting(torch.utils.data.Dataset):
    def __init__(
        self, csv_file: str, input_size: int, forcast_horizon: int, stride: int
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        self.input_size = input_size
        self.forcast_horizon = forcast_horizon
        self.stride = stride

        self.X, self.y = self.run_sliding_window(self.df)

    def run_sliding_window(self, df: pd.DataFrame) -> Tuple[np.array, np.array]:
        """Creates the input-output pairs for the forecasting task.
        Discard windows with NaN values.

        Args:
            df (pd.DataFrame): Time series data.

            Returns:
                X (np.array): Model input data (N, input_size, n_features).
                y (np.array): forecast target data (N, forcast_horizon, n_features).
        """
        timeseries = df.values
        X, y = [], []
        in_st = 0
        out_en = in_st + self.input_size + self.forcast_horizon
        while out_en < len(timeseries):
            in_end = in_st + self.input_size
            out_end = in_end + self.forcast_horizon
            seq_x, seq_y = timeseries[in_st:in_end], timeseries[in_end:out_end]
            if np.isnan(seq_x).any() or np.isnan(seq_y).any():
                in_st += self.stride
                out_en = in_st + self.input_size + self.forcast_horizon
                continue
            X.append(seq_x)
            y.append(seq_y)
            in_st += self.stride
            out_en = in_st + self.input_size + self.forcast_horizon
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.X)


def get_data_loaders(
    csv_file: str,
    input_size: int,
    forcast_horizon: int,
    stride: int,
    batch_size: int,
    valid_set_size: int,
    test_set_size: int,
) -> Tuple[DataLoader, DataLoader]:

    dataset = DatasetForecasting(csv_file, input_size, forcast_horizon, stride)
    train_size = int(len(dataset) * (1 - valid_set_size - test_set_size))
    valid_size = int(len(dataset) * valid_set_size)
    test_size = len(dataset) - train_size - valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader, validloader, testloader


def get_clients_dataloaders(
    data_root: str,
    num_clients: int,
    input_size: int,
    forecast_horizon: int,
    stride: int,
    batch_size: int,
    valid_set_size: int,
    test_set_size: int,
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:

    trainloaders = []
    valloaders = []
    testloaders = []
    assert num_clients <= len(
        glob.glob(data_root + "/*.csv")
    ), "Querying more clients than available"

    for i, csv_file in enumerate(glob.glob(data_root + "/*.csv")):
        trainloader, validloader, testloader = get_data_loaders(
            csv_file,
            input_size,
            forecast_horizon,
            stride,
            batch_size,
            valid_set_size,
            test_set_size,
        )
        trainloaders.append(trainloader)
        valloaders.append(validloader)
        testloaders.append(testloader)
        if i == num_clients - 1:
            break

        if len(trainloader) == 0 or len(validloader) == 0 or len(testloader) == 0:
            raise ValueError(
                f"Client {i} ({csv_file}) has empty dataloaders: len_train={len(trainloader)}, len_valid={len(validloader)}, len_test={len(testloader)}"
            )
            # dataset 11 and 13 have too few samples
            # when its <batch_size, droping last batch (seq2seq requires this) will yield 0 sized batches and errors
            # TODO look into this

    return trainloaders, valloaders, testloaders


if __name__ == "__main__":
    get_clients_dataloaders(  # for debugging
        data_root="data/processed",
        num_clients=20,
        input_size=24 * 6,
        forecast_horizon=24,
        stride=24,
        batch_size=32,
        valid_set_size=0.15,
        test_set_size=0.15,
    )
