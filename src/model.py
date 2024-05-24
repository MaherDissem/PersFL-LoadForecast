from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.SCINet.wrapper import ModelWrapper as SCINet
from models.seq2seq.wrapper import ModelWrapper as Seq2Seq
from config import config
from dataset import get_clients_dataloaders


class ForecastingModel:
    def __init__(
        self,
        config,
        trainloader: DataLoader,
        validloader: DataLoader,
        testloader: DataLoader,
    ):
        self.config = config
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader

        self.len_trainloader = len(trainloader) if trainloader is not None else 0
        self.len_validloader = len(validloader) if validloader is not None else 0
        self.len_testloader = len(testloader) if testloader is not None else 0

        if self.config.model == "SCINet":
            self.model_wrapper = SCINet(
                self.config, self.config.input_size, self.config.forecast_horizon
            )
        elif self.config.model == "Seq2Seq":
            self.model_wrapper = Seq2Seq(
                self.config, self.config.input_size, self.config.forecast_horizon
            )
        else:
            raise NotImplementedError("Model not implemented")

    def train(self) -> Tuple[List[float], float, float, float, float, float]:
        return self.model_wrapper.train(
            self.trainloader, self.validloader, self.testloader
        )

    def evaluate(self) -> Tuple[float, float, float, float, float, float]:
        return self.model_wrapper.validate(self.validloader)

    def test(self) -> Tuple[float, float, float, float, float, float]:
        return self.model_wrapper.validate(self.testloader)

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model_wrapper.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model_wrapper.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [
            val.cpu().numpy()
            for _, val in self.model_wrapper.model.state_dict().items()
        ]


def run_on_local_data(
    trainloader: DataLoader, validloader: DataLoader, testloader: DataLoader
):
    """Train and test a forecasting model on local data only.
    This is used for comparing performance of local training vs federated learning"""

    model = ForecastingModel(
        config=config,
        trainloader=trainloader,
        validloader=validloader,
        testloader=testloader,
    )
    model.train()
    smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = model.test()
    print(
        f"Test Losses: smape={smape_loss:.2f}, mae={mae_loss:.2f}, mse={mse_loss:.2f}, rmse={rmse_loss:.2f}, r2={r2_loss:.2f}"
    )


if __name__ == "__main__":
    cid = 0
    trainloaders, valloaders, testloaders = get_clients_dataloaders(
        data_root=config.data_root,
        num_clients=config.nbr_clients,
        input_size=config.input_size,
        forecast_horizon=config.forecast_horizon,
        stride=config.stride,
        batch_size=config.batch_size,
        valid_set_size=config.valid_set_size,
        test_set_size=config.test_set_size,
    )
    run_on_local_data(trainloaders[cid], valloaders[cid], testloaders[cid])
