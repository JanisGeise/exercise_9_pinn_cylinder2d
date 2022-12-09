"""
    this script contains all shared functions and base classes used for all ODE's
"""
import torch as pt

from typing import Union, Tuple
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


class PINN(pt.nn.Module, ABC):
    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int, n_neurons: int, activation: callable = pt.tanh):
        """
        implements a fully-connected neural network

        :param n_inputs: number of input parameters
        :param n_outputs: number of output parameters
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: none
        """
        super(PINN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.layers = pt.nn.ModuleList()

        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
                self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)

    @abstractmethod
    def compute_loss_equation(self, *args):
        pass

    @abstractmethod
    def compute_loss_prediction(self, *args):
        pass

    @abstractmethod
    def compute_loss_initial_condition(self, *args):
        pass


def train_pinn(model, features_pred, labels_pred, features_eq, labels_eq, epochs: Union[int, float] = 100,
               lr: float = 0.01, save_model: bool = True, save_name: str = "best_model", save_path: str = "",
               equation_params: Union[Tuple, dict, int, float, pt.Tensor, list] = None,
               batch_size: int = 100) -> Tuple[list, list, list, list]:
    """
    train the PINN's

    :param model: the model which should be trained
    :param features_pred: features, namely the time values used for prediction
    :param labels_pred: labels, namely the true x-values of the analytical solution corresponding to the time-values
    :param features_eq: features for evaluating the equation
    :param labels_eq: labels for evaluating the equation
    :param equation_params: all fixed parameters required for solving the equation, e.g. constants or factors
    :param epochs: number of epochs to run the training
    :param lr: learning rate
    :param save_model: flag if the best should be saved
    :param save_name: name of the best model (for saving)
    :param save_path: location where the models should be saved in
    :param batch_size: batch size
    :return: [total loss, equation loss, prediction loss] as tuple of lists
    """

    # optimizer settings
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=(1.0e-4 / 1.0e-2) ** (1.0 / epochs))
    tot_train_loss, pred_loss, eq_loss, init_loss, best_train_loss = [], [], [], [], 1e3

    # create dataset
    dataset_pred = TensorDataset(features_pred, labels_pred)
    dataset_eq = TensorDataset(features_eq, labels_eq)
    dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_eq = DataLoader(dataset_eq, batch_size=batch_size, shuffle=True, drop_last=False)

    for e in range(1, int(epochs) + 1):
        tot_loss_tmp, eq_loss_tmp, pred_loss_tmp, init_loss_tmp = [], [], [], []
        for f_l_pred, f_l_eq in zip(dataloader_pred, dataloader_eq):
            # training loop
            model.train()
            optimizer.zero_grad()

            # loss for prediction: 'compute_loss_prediction'-method always takes features & labels of 'dataloader_pred'
            loss_train_pred = model.compute_loss_prediction(model, f_l_pred[0], f_l_pred[1])

            # loss for equation: 'compute_loss_equation'-method always takes features of 'dataloader_eq' and all fixed
            # parameters of the ODE
            if equation_params is None:
                loss_train_eq = model.compute_loss_equation(model, f_l_eq[0])
                loss_initial_condition = model.compute_loss_initial_condition(model)
            else:
                loss_train_eq = model.compute_loss_equation(model, f_l_eq[0], equation_params)
                loss_initial_condition = model.compute_loss_initial_condition(model, equation_params)

            loss_tot = loss_train_eq + loss_train_pred + loss_initial_condition
            loss_tot.backward()
            optimizer.step()
            tot_loss_tmp.append(loss_tot.item())
            eq_loss_tmp.append(loss_tot.item())
            pred_loss_tmp.append(loss_tot.item())
            init_loss_tmp.append(loss_initial_condition.item())

        tot_train_loss.append(pt.mean(pt.tensor(tot_loss_tmp)))
        eq_loss.append(pt.mean(pt.tensor(eq_loss_tmp)))
        pred_loss.append(pt.mean(pt.tensor(pred_loss_tmp)))
        init_loss.append(pt.mean(pt.tensor(init_loss_tmp)))

        scheduler.step()

        # save best models
        if save_model:
            if tot_train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_path}/{save_name}_train.pt")
                best_train_loss = tot_train_loss[-1]

        # print some info after every 25 epochs
        if e % 100 == 0:
            print(f"finished epoch {e}:\ttraining loss = {round(tot_train_loss[-1].item(), 8)}, \t"
                  f"equation loss = {round(eq_loss[-1].item(), 8)}, "
                  f"\tprediction loss = {round(pred_loss[-1].item(), 8)}",
                  f"\tinitial cond. loss = {round(init_loss[-1].item(), 8)}")

    return tot_train_loss, eq_loss, pred_loss, init_loss


def lhs_sampling(x_min: list, x_max: list, n_samples: int) -> pt.Tensor:
    """
    latin hypercube sampling, taken from:
        https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/ml_intro.ipynb

    :param x_min: lower bounds for each parameter as [min_parameter_1, ... , min_parameter_N]
    :param x_max: upper bounds for each parameter as [max_parameter_1, ... , max_parameter_N]
    :param n_samples: number of points (how many points should be sampled)
    :return: the sampled points for each parameter
    """
    assert len(x_min) == len(x_max)
    n_parameters = len(x_min)
    samples = pt.zeros((n_parameters, n_samples))
    for i, (lower, upper) in enumerate(zip(x_min, x_max)):
        bounds = pt.linspace(lower, upper, n_samples + 1)
        rand = bounds[:-1] + pt.rand(n_samples) * (bounds[1:] - bounds[:-1])
        samples[i, :] = rand[pt.randperm(n_samples)]
    return samples


if __name__ == "__main__":
    pass
