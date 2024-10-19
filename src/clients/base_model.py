from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from forecasting.seq2seq.wrapper import ModelWrapper as Seq2Seq
from forecasting.SCINet.wrapper import ModelWrapper as SCINet
from clients.ALA import ALA
from config import config


class ForecastingModel:
    """This module represents a forecasting model designed for federated learning (no personalization)."""

    def __init__(
        self,
        config: config,
        trainloader: DataLoader,
        validloader: DataLoader,
        testloader: DataLoader,
        cid: int = -1,
    ):
        self.config = config
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader

        self.len_trainloader = len(trainloader) if trainloader is not None else 0
        self.len_validloader = len(validloader) if validloader is not None else 0
        self.len_testloader = len(testloader) if testloader is not None else 0

        self.local_model_wrapper = self.build_model()
        self.ala_initialization = ALA(
            cid=cid,
            loss=self.local_model_wrapper.criterion,
            train_data=self.trainloader,
            batch_size=config.batch_size,
            rand_percent=config.rand_percent,
            layer_idx=config.layer_idx,
            eta=config.eta,
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
            verbose=config.verbose,
        )

    def build_model(self):
        if self.config.model == "SCINet":
            return SCINet(
                self.config, self.config.input_size, self.config.forecast_horizon
            )
        elif self.config.model == "Seq2Seq":
            return Seq2Seq(
                self.config, self.config.input_size, self.config.forecast_horizon
            )
        else:
            raise NotImplementedError("Model not implemented")

    def train(self) -> Tuple[List[float], float, float, float, float, float]:
        return self.local_model_wrapper.train(
            self.trainloader, self.validloader, self.testloader
        )

    def evaluate(self) -> Tuple[float, float, float, float, float, float]:
        return self.local_model_wrapper.validate(self.validloader)

    def test(self) -> Tuple[float, float, float, float, float, float]:
        return self.local_model_wrapper.validate(self.testloader)

    def set_parameters(self, parameters: List[np.ndarray]):
        if not config.ala_init:
            # Initialize local model directly with global model weights (FedAvg)
            params_dict = zip(self.local_model_wrapper.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.local_model_wrapper.model.load_state_dict(state_dict, strict=True)
        else:
            # Initialize local model using global weights through ALA
            # Get global model parameters
            params_dict = zip(self.local_model_wrapper.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            global_model_wrapper = self.build_model()
            global_model_wrapper.model.load_state_dict(state_dict, strict=True)

            # Initialize local model with ALA
            self.ala_initialization.adaptive_local_aggregation(
                global_model_wrapper.model, self.local_model_wrapper.model
            )
\
    def get_parameters(self) -> List[np.ndarray]:
        return [
            val.cpu().numpy()
            for _, val in self.local_model_wrapper.model.state_dict().items()
        ]
