from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from forecasting.seq2seq.wrapper import ModelWrapper as Seq2Seq
from forecasting.SCINet.wrapper import ModelWrapper as SCINet


class ForecastingModel:
    """This module represents a forecasting model designed for federated learning (no personalization)."""

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
