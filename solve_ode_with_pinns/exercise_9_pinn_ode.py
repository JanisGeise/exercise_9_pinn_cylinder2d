"""
    solve simple ODE's using PINN's
    TODO
"""
import torch as pt
import matplotlib.pyplot as plt

from os import path, mkdir
from typing import Union, Tuple
from torch.utils.data import DataLoader, TensorDataset


class PINN(pt.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int, n_neurons: int, activation: callable = pt.relu):
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


def compute_loss(x: pt.Tensor, t: pt.Tensor, k: Union[int, float]) -> pt.Tensor:
    """
    TODO

    :param x:
    :param t:
    :param k:
    :return:
    """
    # compute loss
    mse = pt.nn.MSELoss()

    # ODE: dx/dt + kx = 0
    dx_dt = pt.autograd.grad(outputs=x.sum(), inputs=t, create_graph=True)[0].squeeze()
    loss = mse(dx_dt + k*x, pt.zeros(dx_dt.size()))

    return loss


def compute_loss_pred(x_pred, x_real):
    mse = pt.nn.MSELoss()
    loss = mse(x_pred, x_real)

    return loss


def train_pinn(model, features_train, labels_train, n_epochs: Union[int, float] = 100, lr: float = 0.01,
               k: Union[int, float] = 1, save_model: bool = True, save_name: str = "best_model",
               save_path: str = "", batch_size: int = 25) -> Tuple[list, list, list]:
    """
    TODO

    :param model:
    :param features_train:
    :param labels_train:
    :param n_epochs:
    :param lr:
    :param k:
    :param save_model:
    :param save_name:
    :param save_path:
    :param batch_size:
    :return:
    """

    # optimizer settings
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=(1.0e-4 / 1.0e-2) ** (1.0 / n_epochs))
    tot_train_loss, pred_loss, eq_loss, best_train_loss = [], [], [], 1e3

    # create dataset
    dataset_train = TensorDataset(features_train, labels_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

    for e in range(1, int(n_epochs)+1):
        tot_loss_tmp, eq_loss_tmp, pred_loss_tmp = [], [], []
        for feature, label in dataloader_train:
            # training loop
            model.train()
            optimizer.zero_grad()
            prediction = model(feature).squeeze()
            loss_train_eq = compute_loss(x=prediction, t=feature, k=k)
            loss_train_pred = compute_loss_pred(x_pred=prediction, x_real=label.squeeze())
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
                  f"equation loss = {round(eq_loss[-1].item(), 8)}, \tprediction loss = {round(pred_loss[-1].item(), 8)}")

    return tot_train_loss, eq_loss, pred_loss


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
    load_path = r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/"
    k, n_epochs = 0.25, 1e3

    # create directory for plots
    if not path.exists("".join([load_path, "plots"])):
        mkdir("".join([load_path, "plots"]))

    # ensure reproducibility
    pt.manual_seed(0)

    # instantiate model: we want to predict an x(t) for a given t, but in order to compute dx/dt, at least 2 points are
    # required
    pinn = PINN(n_inputs=1, n_outputs=1, n_layers=5, n_neurons=75)

    # compute analytical solution as comparison
    x, t = compute_analytical_solution(t_end=10, k=k)

    # sample some "real" points from the analytical solution as training- and validation data
    label_train, feature_train = compute_analytical_solution(t_end=10, n_points=10, k=k)

    # for some reason normalize data makes it really bad...
    # x_pred, t_pred = compute_analytical_solution(n_points=100, k=k)
    # label_train, min_max_x = scale_data(x_pred)
    # feature_train, min_max_t = scale_data(t_pred)

    # train model using the sampled points
    losses = train_pinn(pinn, feature_train.unsqueeze(-1).requires_grad_(True),
                        label_train.unsqueeze(-1).requires_grad_(True), k=k, n_epochs=n_epochs, save_path=load_path)

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

    # rescale data
    # x_pred = rescale_data(x_pred, min_max_x)

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
