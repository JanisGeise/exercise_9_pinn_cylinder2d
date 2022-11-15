"""
    training routine for PINN training

    Note: this code is a modified version of the code presented in this repository:
          https://github.com/Shengfeng233/PINN-for-NS-equation
"""
from os import mkdir
from os.path import exists
from time import time

import torch as pt
import numpy as np
from torch import nn
from pandas import DataFrame


class PINN(nn.Module):
    """
    this class implements a PINN
    """
    def __init__(self, n_layers: int = 10, n_neurons: int = 50):
        super(PINN, self).__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.base = nn.Sequential()

        # input layer to first hidden layer
        self.base.add_module("0linear", nn.Linear(3, n_neurons))
        self.base.add_module("0Act", nn.Tanh())

        # add more hidden layers if specified
        if self.n_layers > 1:
            for i in range(1, self.n_layers - 1):
                self.base.add_module(str(i) + "linear", nn.Linear(self.n_neurons, self.n_neurons))
                self.base.add_module(str(i) + "Act", nn.Tanh())

        # last hidden layer to output
        self.base.add_module(str(self.n_layers) + "linear", nn.Linear(self.n_neurons, 2))
        self.lam1 = nn.Parameter(pt.randn(1, requires_grad=True))
        self.lam2 = nn.Parameter(pt.randn(1, requires_grad=True))
        self.Initial_param()

    def forward(self, x, y, t):
        X = pt.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # Initialize the parameters
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

    # In-class method: find the loss of data points
    def data_mse(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        p_predict = predict_out[:, 1].reshape(-1, 1)
        u_predict = pt.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -pt.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = pt.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # In-class method: find the loss of data points (excluding pressure data)
    def data_mse_without_p(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = pt.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -pt.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = pt.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # In-class method: find the loss of the equation point
    def equation_mse(self, x, y, t, Re):
        # Correct the problem, the user needs to provide the parameter values of the system by himself,
        # the default is 1&0.01
        predict_out = self.forward(x, y, t)

        # Get the predicted output psi, p
        psi = predict_out[:, 0].reshape(-1, 1)
        p = predict_out[:, 1].reshape(-1, 1)

        # Calculate each partial derivative by automatic differentiation, where.sum() converts a vector into a scalar
        u = pt.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v = -pt.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u_t = pt.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = pt.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = pt.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_t = pt.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = pt.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = pt.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = pt.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = pt.autograd.grad(p.sum(), y, create_graph=True)[0]
        u_xx = pt.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = pt.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = pt.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = pt.autograd.grad(v_y.sum(), y, create_graph=True)[0]

        # Calculate the residual of partial differential equations
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0/Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0/Re * (v_xx + v_yy)
        mse = pt.nn.MSELoss()
        batch_t_zeros = (pt.zeros((x.shape[0], 1))).float().requires_grad_(True)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)
        return mse_equation


def f_equation_inverse(x, y, t, model):
    """
    Define the inverse of the partial differential equation as the inverse problem in order to make predictions

    :param x: x-coordinates
    :param y: y-coordinates
    :param t: time series
    :param model: trained model
    :return:
    """
    # lam1 = model.lam1
    # lam2 = model.lam2
    predict_out = model.forward(x, y, t)

    # Get the predicted output psi, p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)

    # Calculate each partial derivative by automatic differentiation, where.sum() converts a vector into a scalar
    u = pt.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -pt.autograd.grad(psi.sum(), x, create_graph=True)[0]
    """
    for some reason, the following is not used in prediction-fct:

    u_t = pt.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = pt.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = pt.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = pt.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = pt.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = pt.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = pt.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = pt.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = pt.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = pt.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = pt.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = pt.autograd.grad(v_y.sum(), y, create_graph=True)[0]

    # Calculate the residual of partial differential equations
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    """
    return u, v, p


def shuffle_data(x, y, t, u, v, p):
    x_total = pt.cat([x, y, t, u, v, p], 1)
    x_total_arr = x_total.data.numpy()
    np.random.shuffle(x_total_arr)
    return pt.tensor(x_total_arr)


def train_pinns(inner_iter: int, x_random: pt.Tensor, batch_size_data: int, batch_size_eqa: int, eq_points: int,
                re_no: int = 100, epochs: int = 1000, name_model: str = "model_train", path: str = "./test_training/",
                name_loss: str = "loss") -> None:

    if not exists(path):
        mkdir(path)

    best_loss = 1.0e5
    pinn_net = PINN()
    losses = np.empty((0, 3), dtype=float)
    optimizer = pt.optim.Adam(pinn_net.parameters(), lr=0.001)
    scheduler = pt.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # start timer
    start = time()
    for epoch in range(epochs):
        for batch_iter in range(inner_iter + 1):
            optimizer.zero_grad()
            # Randomly take batch in the complete works
            if batch_iter < inner_iter:
                x_train = x_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 0].reshape(
                    batch_size_data, 1).clone().requires_grad_(True)
                y_train = x_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 1].reshape(
                    batch_size_data, 1).clone().requires_grad_(True)
                t_train = x_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 2].reshape(
                    batch_size_data, 1).clone().requires_grad_(True)
                u_train = x_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 3].reshape(
                    batch_size_data, 1).clone().requires_grad_(True)
                v_train = x_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 4].reshape(
                    batch_size_data, 1).clone().requires_grad_(True)
                x_eqa = eq_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 0].reshape(
                    batch_size_eqa, 1).clone().requires_grad_(True)
                y_eqa = eq_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 1].reshape(
                    batch_size_eqa, 1).clone().requires_grad_(True)
                t_eqa = eq_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 2].reshape(
                    batch_size_eqa, 1).clone().requires_grad_(True)
            elif batch_iter == inner_iter:
                if x_random[batch_iter * batch_size_data:, 0].reshape(-1, 1).shape[0] == 0:
                    continue
                else:
                    x_train = x_random[batch_iter * batch_size_data:, 0].reshape(-1, 1).clone().requires_grad_(True)
                    y_train = x_random[batch_iter * batch_size_data:, 1].reshape(-1, 1).clone().requires_grad_(True)
                    t_train = x_random[batch_iter * batch_size_data:, 2].reshape(-1, 1).clone().requires_grad_(True)
                    u_train = x_random[batch_iter * batch_size_data:, 3].reshape(-1, 1).clone().requires_grad_(True)
                    v_train = x_random[batch_iter * batch_size_data:, 4].reshape(-1, 1).clone().requires_grad_(True)
                    x_eqa = eq_points[batch_iter * batch_size_eqa:, 0].reshape(-1, 1).clone().requires_grad_(True)
                    y_eqa = eq_points[batch_iter * batch_size_eqa:, 1].reshape(-1, 1).clone().requires_grad_(True)
                    t_eqa = eq_points[batch_iter * batch_size_eqa:, 2].reshape(-1, 1).clone().requires_grad_(True)

            mse_predict = pinn_net.data_mse_without_p(x_train, y_train, t_train, u_train, v_train)
            mse_equation = pinn_net.equation_mse(x_eqa, y_eqa, t_eqa, re_no)

            # Calculate the loss function without introducing the true value of the pressure field
            loss = mse_predict + mse_equation
            loss.backward()
            optimizer.step()
            with pt.autograd.no_grad():
                # Output status
                if (batch_iter + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}\tBatch iter: {batch_iter+1}\t\tTraining loss: {round(float(loss.data), 8)}")

                if (batch_iter + 1) % inner_iter == 0:
                    loss_all = loss.cpu().data.numpy().reshape(1, 1)
                    loss_predict = mse_predict.cpu().data.numpy().reshape(1, 1)
                    loss_equation = mse_equation.cpu().data.numpy().reshape(1, 1)
                    loss_set = np.concatenate((loss_all, loss_predict, loss_equation), 1)
                    losses = np.append(losses, loss_set, 0)
                    loss_save = DataFrame(losses)
                    loss_save.to_csv("".join([path, name_loss, ".csv"]), index=False, header=False)
                    del loss_save

                # save best model
                if float(loss.data) < best_loss:
                    pt.save(pinn_net.state_dict(), "".join([path, name_model, ".pt"]))
                    best_loss = float(loss.data)
        scheduler.step()

    print(f"Training took {time() - start} s")


if __name__ == "__main__":
    pass
