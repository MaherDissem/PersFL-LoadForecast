from typing import List, Tuple
from collections import OrderedDict
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.seq2seq.wrapper import ModelWrapper as Seq2Seq
from models.SCINet.wrapper import ModelWrapper as SCINet
from models.SCINet.wrapper import smooth_l1_loss, adjust_learning_rate, smooth_l1_loss
from models.early_stop import EarlyStopping
from dataset import get_clients_dataloaders
from config import config
from utils import set_seed


class ForecastingModel(nn.Module):
    """This module represents a forecasting model that mixes a local model with a federated one to achieve personalization."""

    def __init__(
        self,
        config,
        trainloader: DataLoader,
        validloader: DataLoader,
        testloader: DataLoader,
    ):
        super(ForecastingModel, self).__init__()
        self.args = config
        set_seed(self.args.seed)
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader

        self.len_trainloader = len(trainloader) if trainloader is not None else 0
        self.len_validloader = len(validloader) if validloader is not None else 0
        self.len_testloader = len(testloader) if testloader is not None else 0

        self.model_m = self.build_model()  # mixed model
        self.model_l = self.build_model()  # local model (private)
        self.model_f = self.build_model()  # federated model

        self.optimizer_m = self.get_optimizer(self.model_m.parameters())
        self.optimizer_l = self.get_optimizer(self.model_l.parameters())
        self.optimizer_f = self.get_optimizer(self.model_f.parameters())

        self.criterion = self.get_criterion()

    def build_model(self) -> torch.nn.Module:
        # These are wrappers over the model with their own train and validate methods, but we'll make new ones to implement the model-mixing logic
        if self.args.model == "SCINet":
            return SCINet(
                self.args, self.args.input_size, self.args.forecast_horizon
            ).model
        elif self.args.model == "Seq2Seq":
            return Seq2Seq(
                self.args, self.args.input_size, self.args.forecast_horizon
            ).model
        else:
            raise NotImplementedError("Model not implemented")

    def get_optimizer(self, parameters: List[torch.nn.Parameter]):
        return torch.optim.Adam(
            params=parameters,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

    def get_criterion(self):
        return (
            smooth_l1_loss
            if self.args.L1Loss
            else nn.MSELoss(size_average=False).cuda()
        )

    def _sample_alpha(self) -> float:
        """Returns alpha sampled from Uniform(0,1)"""
        alpha = np.random.uniform()
        self.alpha = alpha
        return alpha

    def forward(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """Randomly mixes the local and federated models and perform inference."""
        if alpha is None:
            alpha = self._sample_alpha()
        # Interpolate the parameters of model1 and model2
        for param_l, param_f, param_m in zip(
            self.model_l.parameters(),
            self.model_f.parameters(),
            self.model_m.parameters(),
        ):
            param_m.data = (1 - alpha) * param_f.data + alpha * param_l.data

        return self.model_m(x)

    def backward(self, loss: torch.Tensor):
        """Backpropagate the loss and update the models' weights."""
        # Clear the gradients
        self.optimizer_m.zero_grad()
        self.optimizer_l.zero_grad()
        self.optimizer_f.zero_grad()

        # Backpropagate the loss
        loss.backward()
        self.optimizer_m.step()

        # Update the original models weights (frozen) with the mixed model's learned weights
        # model_l: grad_l = alpha * grad_m.
        # model_f: grad_f = (1-alpha) * grad_m.
        # This is easy to prove.
        with torch.no_grad():
            for param_l, param_f, param_m in zip(
                self.model_l.parameters(),
                self.model_f.parameters(),
                self.model_m.parameters(),
            ):
                param_l.grad = self.alpha * param_m.grad
                param_f.grad = (1 - self.alpha) * param_m.grad

        self.optimizer_l.step()
        self.optimizer_f.step()

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set client model parameters from server model parameters."""
        # Load local model from saved checkpoint
        if os.path.exists(self.args.checkpoint_path):  # doesn't exist at server side
            self.load_parameters(torch.load(self.args.checkpoint_path))
            #TODO save and load optimal local model weights
        # Load federated model from server
        params_dict = zip(self.model_f.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model_f.load_state_dict(state_dict, strict=True)

        # torch.save(parameters, "aggregated_parameters.pth") # TODO remove this debug line

    def get_parameters(self) -> List[np.ndarray]:
        """Send federated client model parameters to server."""
        return [val.cpu().numpy() for _, val in self.model_f.state_dict().items()]

    def load_parameters(self, state_dict: OrderedDict):
        """Loads mixed, local and federated model weights from a state dict."""
        model_m_state_dict = {}
        model_l_state_dict = {}
        model_f_state_dict = {}

        for name, param in state_dict.items():
            if "model_m." in name:
                name = name.replace("model_m.", "")
                model_m_state_dict[name] = param
            elif "model_l." in name:
                name = name.replace("model_l.", "")
                model_l_state_dict[name] = param
            elif "model_f." in name:
                name = name.replace("model_f.", "")
                model_f_state_dict[name] = param

        self.model_m.load_state_dict(model_m_state_dict, strict=True)
        self.model_l.load_state_dict(model_l_state_dict, strict=True)
        self.model_f.load_state_dict(model_f_state_dict, strict=True)

    def train(self) -> Tuple[List[float], float, float, float, float, float]:
        return self._train(self.trainloader, self.validloader, self.testloader)

    def evaluate(self) -> Tuple[float, float, float, float, float, float]:
        return self._validate(self.validloader)

    def test(
        self, server_side=False
    ) -> Tuple[float, float, float, float, float, float]:
        if server_side:
            return self._validate(self.testloader, self.model_f)
        elif self.args.eval_local:
            return self._validate(self.testloader, self.model_l)
        else:
            return self._validate(self.testloader, self.model_m)

    def _train(
        self, trainloader: DataLoader, validloader: DataLoader, testloader: DataLoader
    ) -> Tuple[List[float], float, float, float, float, float]:

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            checkpoint_path=self.args.checkpoint_path,
            verbose=False,
        )

        epoch_start = 0
        loss_evol = []

        if self.args.mu > 0:
            model_g = copy.deepcopy(
                self.model_f
            )  # global model (frozen federated model)
            for param in model_g.parameters():
                param.requires_grad = False

        for epoch in range(epoch_start, self.args.epochs):
            self.model_m.train()  # controls behavior of dropout and batchnorm
            epoch_loss = 0.0
            adjust_learning_rate(self.optimizer_m, epoch, self.args)

            for data in trainloader:
                inputs, targets = data
                inputs = inputs.to(self.device)  # [batch_size, seq_len, n_var]
                targets = targets.to(self.device)  # [batch_size, horizon, n_var]

                # Inference and criterion loss
                self.zero_grad()
                if self.args.stacks == 1:
                    forecast = self(inputs)
                    loss = self.criterion(forecast, targets)
                if self.args.stacks == 2:
                    forecast, mid = self(inputs)
                    loss = self.criterion(forecast, targets) + self.criterion(
                        mid, targets
                    )

                # Proximity regularization loss
                if self.args.mu > 0:
                    prox = 0.0
                    for param_f, param_g in zip(
                        self.model_f.parameters(), model_g.parameters()
                    ):
                        prox += (param_f - param_g).norm(2)
                    loss += self.args.mu * prox

                # Subspace construciton loss
                numerator, norm_1, norm_2 = 0, 0, 0
                for param_f, param_l in zip(
                    self.model_f.parameters(), self.model_l.parameters()
                ):
                    numerator += (param_f * param_l).add(1e-6).sum()
                    norm_1 += param_f.pow(2).sum()
                    norm_2 += param_l.pow(2).sum()
                cos_sim = numerator.pow(2).div(norm_1 * norm_2)
                loss += self.args.nu * cos_sim

                epoch_loss += loss.item()

                # Backward pass
                self.backward(loss)

            epoch_loss /= len(trainloader)  # average loss per batch
            loss_evol.append(epoch_loss)  # keeps track of loss evolution

            # valid
            smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self._validate(
                validloader, self.model_m
            )

            if self.args.verbose:
                print(
                    f"Epoch {epoch}: train loss={epoch_loss:.2f}, valid loss={smape_loss:.2f}"
                )

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(mse_loss, self)

            if early_stopping.early_stop:
                break

        # load the last checkpoint with the best model (saved by EarlyStopping)
        saved_state_dict = torch.load(self.args.checkpoint_path)
        self.load_parameters(saved_state_dict)

        # Eval local model on test set
        if self.args.eval_local:
            smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self._validate(
                testloader, self.model_l
            )
        # Eval different models from the low loss subspace on valid set then test best one on test set (Superfed paper)
        else:
            results = []
            for alpha in np.arange(0, 1.1, 0.1):
                smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self._validate(
                    validloader, alpha=alpha
                )
                results.append(
                    (alpha, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss)
                )
            best_alpha, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = min(
                results, key=lambda x: x[1]
            )
            smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = self._validate(
                testloader, alpha=best_alpha
            )
        return loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss

    def _validate(
        self, dataloader: DataLoader, model: nn.Module = None, alpha: float = None
    ) -> Tuple[float, float, float, float, float]:
        
        losses_smape = []
        losses_mae = []
        losses_mse = []
        losses_rmse = []
        losses_r2 = []

        for _data in dataloader:
            inputs, targets = _data
            inputs = inputs.to(self.device)  # [batch_size, window_size, n_var]
            targets = targets.to(self.device)  # [batch_size, horizon, n_var]
            with torch.no_grad():
                if model is not None:
                    model.eval()
                    if self.args.stacks == 1:
                        outputs = model(inputs)
                    elif self.args.stacks == 2:
                        outputs, _ = model(inputs)
                else:
                    if self.args.stacks == 1:
                        outputs = self(inputs, alpha)
                    elif self.args.stacks == 2:
                        outputs, _ = self(inputs, alpha)
            # sMAPE
            absolute_percentage_errors = (
                2
                * torch.abs(outputs - targets)
                / (torch.abs(outputs) + torch.abs(targets))
            )
            loss_smape = torch.mean(absolute_percentage_errors) * 100
            # MAE
            loss_mae = torch.mean(torch.abs(outputs - targets))
            # MSE
            loss_mse = torch.mean((outputs - targets) ** 2)
            # RMSE
            loss_rmse = torch.sqrt(loss_mse)
            # R squared
            loss_r2 = 1 - torch.sum((targets - outputs) ** 2) / torch.sum(
                (targets - torch.mean(targets)) ** 2
            )

            losses_smape.append(loss_smape.item())
            losses_mae.append(loss_mae.item())
            losses_mse.append(loss_mse.item())
            losses_rmse.append(loss_rmse.item())
            losses_r2.append(loss_r2.item())

        smape_loss = np.array(losses_smape).mean()
        mae_loss = np.array(losses_mae).mean()
        mse_loss = np.array(losses_mse).mean()
        rmse_loss = np.array(losses_rmse).mean()
        r2_loss = np.array(losses_r2).mean()

        return smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# debug


# def validate(model, dataloader):
#     model.eval()
#     device = (
#         torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#     )

#     losses_smape = []
#     losses_mae = []
#     losses_mse = []
#     losses_rmse = []
#     losses_r2 = []

#     for _data in dataloader:
#         inputs, targets = _data
#         inputs = inputs.to(device)  # [batch_size, window_size, n_var]
#         targets = targets.to(device)  # [batch_size, horizon, n_var]
#         with torch.no_grad():
#             if config.stacks == 1:
#                 outputs = model(inputs)
#             elif config.stacks == 2:
#                 outputs, _ = model(inputs)

#         # sMAPE
#         absolute_percentage_errors = (
#             2 * torch.abs(outputs - targets) / (torch.abs(outputs) + torch.abs(targets))
#         )
#         loss_smape = torch.mean(absolute_percentage_errors) * 100
#         # MAE
#         loss_mae = torch.mean(torch.abs(outputs - targets))
#         # MSE
#         loss_mse = torch.mean((outputs - targets) ** 2)
#         # RMSE
#         loss_rmse = torch.sqrt(loss_mse)
#         # R squared
#         loss_r2 = 1 - torch.sum((targets - outputs) ** 2) / torch.sum(
#             (targets - torch.mean(targets)) ** 2
#         )

#         losses_smape.append(loss_smape.item())
#         losses_mae.append(loss_mae.item())
#         losses_mse.append(loss_mse.item())
#         losses_rmse.append(loss_rmse.item())
#         losses_r2.append(loss_r2.item())

#     smape_loss = np.array(losses_smape).mean()
#     mae_loss = np.array(losses_mae).mean()
#     mse_loss = np.array(losses_mse).mean()
#     rmse_loss = np.array(losses_rmse).mean()
#     r2_loss = np.array(losses_r2).mean()

#     return smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


# set_seed(config.seed)
# trainloaders, valloaders, testloaders = get_clients_dataloaders(
#     data_root=config.data_root,
#     num_clients=config.nbr_clients,
#     input_size=config.input_size,
#     forecast_horizon=config.forecast_horizon,
#     stride=config.stride,
#     batch_size=config.batch_size,
#     valid_set_size=config.valid_set_size,
#     test_set_size=config.test_set_size,
# )

# main_model = ForecastingModel(config, trainloaders[0], valloaders[0], testloaders[0])

# # debug
# # main_model.load_parameters(torch.load("weights/model_0 - Copy.pth"))

# # load aggregated parameters
# main_model.set_parameters(torch.load("C:\\Users\\maher\\SimpleFL\\aggregated_parameters.pth"))

# # valid init models
# init_fed_model = main_model.model_f
# init_loc_model = main_model.model_l
# init_smape_f, init_mae_f, init_mse_f, init_rmse_f, init_r2_f = validate(
#     init_fed_model, valloaders[0]
# )
# init_smape_l, init_mae_l, init_mse_l, init_rmse_l, init_r2_l = validate(
#     init_loc_model, valloaders[0]
# )

# # train
# loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = main_model.train()

# # new trained weights
# new_fed_model = main_model.model_f
# new_loc_model = main_model.model_l
# new_smape_f, new_mae_f, new_mse_f, new_rmse_f, new_r2_f = validate(
#     new_fed_model, valloaders[0]
# )
# new_smape_l, new_mae_l, new_mse_l, new_rmse_l, new_r2_l = validate(
#     new_loc_model, valloaders[0]
# )

# print("\nmain model:")
# print(smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss)

# print("\ninit model:")
# print(init_smape_f, init_mae_f, init_mse_f, init_rmse_f, init_r2_f)
# print(init_smape_l, init_mae_l, init_mse_l, init_rmse_l, init_r2_l)

# print("\nnew model:")
# print(new_smape_f, new_mae_f, new_mse_f, new_rmse_f, new_r2_f)
# print(new_smape_l, new_mae_l, new_mse_l, new_rmse_l, new_r2_l)


# ====================================================================================================
# ====================================================================================================
# ====================================================================================================


def run_on_local_data(
    trainloader: DataLoader, validloader: DataLoader, testloader: DataLoader
):
    """Train and test a forecasting model on local data only.
    This is used for comparing performance of local training vs federated learning"""

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


if __name__ == "__main__":
    cid = 0
    set_seed(config.seed + cid)
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
    loss_evol, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = run_on_local_data(
        trainloaders[cid], valloaders[cid], testloaders[cid]
    )
    print(
        f"Test Losses: smape={smape_loss:.2f}, mae={mae_loss:.2f}, mse={mse_loss:.2f}, rmse={rmse_loss:.2f}, r2={r2_loss:.2f}"
    )
    # TODO loop over config.nbr_clients and log results
