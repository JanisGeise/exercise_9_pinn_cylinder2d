"""
    solve ODE's using PINN's

    1. test case:
        dx/dt = -kx     with    x(t=0) = 1, k = const.

"""
import torch as pt
import matplotlib.pyplot as plt

from os import path, mkdir
from typing import Union, Tuple
from torch.utils.data import DataLoader, TensorDataset


class PINN(pt.nn.Module):
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

    def compute_loss_equation(self, t: pt.Tensor, k: Union[int, float]) -> pt.Tensor:
        """
        computes the MSE loss of the ODE using the predicted x-value for a given t-value

        :param t: points in time
        :param t: time-value used for predicting the x-value
        :param k: decay factor
        :return: MSE-loss
        """
        # make prediction
        out = pinn.forward(t).squeeze()

        # compute loss
        mse = pt.nn.MSELoss()

        # ODE: dx/dt + kx = 0
        dx_dt = pt.autograd.grad(outputs=out, inputs=t, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()
        loss = mse(dx_dt + k*out, pt.zeros(dx_dt.size()))

        return loss

    def compute_loss_prediction(self, t: pt.Tensor, x_real: pt.Tensor) -> pt.Tensor:
        """
        computes the prediction loss, meaning the loss between true x-value (label) and predicted x

        :param t: points in time
        :param x_real: true x-value
        :return:  MSE loss between true x-value and prediction
        """
        # make prediction
        out = pinn.forward(t).squeeze()

        mse = pt.nn.MSELoss()
        loss = mse(out, x_real)

        return loss


def train_pinn(model, features_pred, labels_pred, features_eq, labels_eq, epochs: Union[int, float] = 100,
               lr: float = 0.01, k: Union[int, float] = 1, save_model: bool = True, save_name: str = "best_model",
               save_path: str = "", batch_size: int = 50) -> Tuple[list, list, list]:
    """
    train the PINN's

    :param model: the model which should be trained
    :param features_pred: features, namely the time values used for prediction
    :param labels_pred: labels, namely the true x-values of the analytical solution corresponding to the time-values
    :param features_eq: features for evaluating the equation
    :param labels_eq: labels for evaluating the equation
    :param epochs: number of epochs to run the training
    :param lr: learning rate
    :param k: decay factor in the exponent of the ODE
    :param save_model: flag if the best should be saved
    :param save_name: name of the best model (for saving)
    :param save_path: location where the models should be saved in
    :param batch_size: batch size
    :return: [total loss, equation loss, prediction loss] as tuple of lists
    """

    # optimizer settings
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=(1.0e-4 / 1.0e-2) ** (1.0 / n_epochs))
    tot_train_loss, pred_loss, eq_loss, best_train_loss = [], [], [], 1e3

    # create dataset
    dataset_pred = TensorDataset(features_pred, labels_pred)
    dataset_eq = TensorDataset(features_eq, labels_eq)
    dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_eq = DataLoader(dataset_eq, batch_size=batch_size, shuffle=True, drop_last=False)

    for e in range(1, int(epochs)+1):
        tot_loss_tmp, eq_loss_tmp, pred_loss_tmp = [], [], []
        for f_l_pred, f_l_eq in zip(dataloader_pred, dataloader_eq):
            # training loop
            model.train()
            optimizer.zero_grad()

            # loss for prediction
            loss_train_pred = model.compute_loss_prediction(t=f_l_pred[0].squeeze(0), x_real=f_l_pred[1].squeeze())

            # loss for equation
            loss_train_eq = model.compute_loss_equation(t=f_l_eq[0].squeeze(0), k=k)

            loss_tot = loss_train_eq + loss_train_pred
            loss_tot.backward()
            optimizer.step()
            tot_loss_tmp.append(loss_tot.item())
            eq_loss_tmp.append(loss_tot.item())
            pred_loss_tmp.append(loss_tot.item())

        tot_train_loss.append(pt.mean(pt.tensor(tot_loss_tmp)))
        eq_loss.append(pt.mean(pt.tensor(eq_loss_tmp)))
        pred_loss.append(pt.mean(pt.tensor(pred_loss_tmp)))

        scheduler.step()

        # save best models
        if save_model:
            if tot_train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_path}/{save_name}_train.pt")
                best_train_loss = tot_train_loss[-1]

        # print some info after every 25 epochs
        if e % 100 == 0:
            print(f"finished epoch {e},\ttraining loss = {round(tot_train_loss[-1].item(), 8)}, \t"
                  f"equation loss = {round(eq_loss[-1].item(), 8)}, "
                  f"\tprediction loss = {round(pred_loss[-1].item(), 8)}")

    return tot_train_loss, eq_loss, pred_loss


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
        bounds = pt.linspace(lower, upper, n_samples+1)
        rand = bounds[:-1] + pt.rand(n_samples) * (bounds[1:]-bounds[:-1])
        samples[i, :] = rand[pt.randperm(n_samples)]
    return samples


def compute_analytical_solution(t_start: int = 0, t_end: int = 1, k: Union[int, float] = 1, c: int = 0,
                                n_points: int = 1000) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    computes the analytical solution to the ODE
        dx/dt = -kx;    x(t=0) = 1, k = const.

    :param t_start: start time
    :param t_end: end time
    :param k: value for parameter "k" (factor for decay rate)
    :param n_points: number of points for which the solution should be calculated
    :param c: initial condition for x(t=0)
    :return: x(t) and corresponding t as tensor
    """
    # analytical solution: x(t) = exp(-kt)
    t = pt.linspace(t_start, t_end, n_points)
    x = pt.exp(-k*t) + c

    return x, t


def scale_data(x: pt.Tensor) -> Tuple[pt.Tensor, list]:
    """
    normalize data to the interval [0, 1] using a min-max-normalization

    :param x: data which should be normalized
    :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
    """
    # x_i_normalized = (x_i - x_min) / (x_max - x_min)
    x_min_max = [pt.min(x), pt.max(x)]
    return pt.sub(x, x_min_max[0]) / (x_min_max[1] - x_min_max[0]), x_min_max


def rescale_data(x: pt.Tensor, x_min_max: list) -> pt.Tensor:
    """
    reverse the normalization of the data

    :param x: normalized data
    :param x_min_max: min- and max-value used for normalizing the data
    :return: de-normalized data as tensor
    """
    # x = (x_max - x_min) * x_norm + x_min
    return (x_min_max[1] - x_min_max[0]) * x + x_min_max[0]


if __name__ == "__main__":
    # setup
    load_path = r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/"
    k, n_epochs = 0.25, 500

    # create directory for plots
    if not path.exists("".join([load_path, "plots"])):
        mkdir("".join([load_path, "plots"]))

    # ensure reproducibility
    pt.manual_seed(0)

    # instantiate model: we want to predict an x(t) for a given t
    pinn = PINN(n_inputs=1, n_outputs=1, n_layers=5, n_neurons=75)

    # compute analytical solution as comparison
    x, t = compute_analytical_solution(t_end=10, k=k)

    # sample some "real" points from the analytical solution as training- and validation data
    label_pred, feature_pred = compute_analytical_solution(t_end=10, n_points=5, k=k)

    # use latin hypercube sampling to sample points as feature-label pairs for prediction (still not working...)
    # feature_pred = lhs_sampling([0], [10], 5)
    # label_pred = pt.exp(-k * feature_pred)

    # sample points for which the equation should be evaluated
    feature_eq = lhs_sampling([0], [10], 50)
    label_eq = pt.exp(-k * feature_eq)

    # sanity check
    plt.plot(t, x, color="black", label="analytical solution")
    plt.plot(feature_pred, label_pred, color="red", linestyle="none", marker="o")
    plt.plot(feature_eq, label_eq, color="green", linestyle="none", marker="x")

    # dummy points for legend
    plt.plot(feature_pred[0], label_pred[0], color="red", linestyle="none", marker="o", label="points prediction")
    plt.plot(feature_eq[0], label_eq[0], color="green", linestyle="none", marker="x", label="points equation")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([load_path, f"/plots/sampled_points.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # train model using the sampled points
    losses = train_pinn(pinn, feature_pred.unsqueeze(-1).requires_grad_(True),
                        label_pred.unsqueeze(-1).requires_grad_(True), feature_eq.unsqueeze(-1).requires_grad_(True),
                        label_eq.unsqueeze(-1).requires_grad_(True), k=k, epochs=n_epochs, save_path=load_path)

    # plot training- and validation losses
    plt.plot(range(int(n_epochs)), losses[0], color="blue", label="total training loss")
    plt.plot(range(int(n_epochs)), losses[1], color="green", label="equation loss")
    plt.plot(range(int(n_epochs)), losses[2], color="red", label="prediction loss")
    plt.xlabel("$epoch$ $number$", usetex=True, fontsize=14)
    plt.ylabel("$MSE$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncols=1)
    plt.yscale("log")
    plt.savefig("".join([load_path, f"/plots/losses.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

    # load the best model and set to eval() mode
    pinn.load_state_dict(pt.load(load_path + "best_model_train.pt"))
    pinn.eval()

    # predict the solution x(t) based on given t
    x_pred = pt.zeros(x.size())

    # initial condition
    x_pred[0] = 1
    for time in range(1, len(t)):
        x_pred[time] = pinn(t[time].unsqueeze(-1)).detach().squeeze()

    # plot analytical solution vs. predicted one
    plt.plot(t, x, color="black", label="analytical solution")
    plt.plot(t, x_pred, color="red", label="predicted solution")
    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([load_path, f"/plots/prediction_vs_analytical_solution.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")
