"""
    TODO
"""
import os
import pickle
from os.path import exists

import torch as pt
from typing import Tuple

from matplotlib import pyplot as plt
from torch import manual_seed, Tensor
from torch.utils.data import DataLoader, TensorDataset

from post_process_results import compare_flow_fields, make_flow_gif
from dataloader import scale_data


class FCModel(pt.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_layers: int = 3, n_neurons: int = 25,
                 activation: callable = pt.tanh):
        """
        implements a fully-connected neural network

        :param n_inputs: x-, y- and t
        :param n_outputs: psi & p
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: none
        """
        super(FCModel, self).__init__()
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

    def loss_prediction(self, feature, len_x, len_y, u, v) -> pt.Tensor:
        # in order to use autograd, the input variables need to have the same name as used when derivate
        x, y, t = feature[:, :len_x], feature[:, len_x:len_x + len_y], feature[:, -1]

        # calculate MSE loss of velocity fields
        out = self.forward(pt.concat([x, y, t.unsqueeze(-1)], dim=1))
        psi = out[:, :len_x]

        # derivatives: u = d(Psi) / dy, v = - d(Psi) / dx
        u_predict = pt.autograd.grad(psi, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_predict = -pt.autograd.grad(psi, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        mse = pt.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    def loss_equation(self, feature, len_x, len_y, Re: int = 100) -> pt.Tensor:
        # in order to use autograd, the input variables need to have the same name as used when derivate
        x, y, t = feature[:, :len_x], feature[:, len_x:len_x + len_y], feature[:, -1]

        # calculate MSE loss of the NS-equations, predict Psi and p for a given x, y and t
        predict_out = self.forward(pt.concat([x, y, t.unsqueeze(-1)], dim=1))
        psi = predict_out[:, :len_x].squeeze()
        p = predict_out[:, len_x:].squeeze()

        # Calculate each partial derivative by automatic differentiation
        u = pt.autograd.grad(psi, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v = -pt.autograd.grad(psi, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        u_t = pt.autograd.grad(u, t, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        u_x = pt.autograd.grad(u, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        u_y = pt.autograd.grad(u, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_t = pt.autograd.grad(v, t, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_x = pt.autograd.grad(v, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_y = pt.autograd.grad(v, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        p_x = pt.autograd.grad(p, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        p_y = pt.autograd.grad(p, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        u_xx = pt.autograd.grad(u_x, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        u_yy = pt.autograd.grad(u_y, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_xx = pt.autograd.grad(v_x, x, create_graph=True, grad_outputs=pt.ones_like(psi))[0]
        v_yy = pt.autograd.grad(v_y, y, create_graph=True, grad_outputs=pt.ones_like(psi))[0]

        # non-dimensionalized NS-equations
        f = u_t.unsqueeze(-1) * pt.ones(u.size()) + (u * u_x + v * u_y) + p_x - 1.0/Re * (u_xx + u_yy)
        g = v_t.unsqueeze(-1) * pt.ones(u.size()) + (u * v_x + v * v_y) + p_y - 1.0/Re * (v_xx + v_yy)

        # equations = 0 -> MSE = difference to zero
        mse = pt.nn.MSELoss()
        mse_equation = mse(f, pt.zeros(f.size())) + mse(g, pt.zeros(g.size()))
        return mse_equation


def train_model(model: pt.nn.Module, features_train: pt.Tensor, labels_train: pt.Tensor, x: pt.Tensor, y: pt.Tensor,
                re_no: int = 100, epochs: int = 1000, lr: float = 0.01, batch_size: int = 2, save_model: bool = True,
                save_name: str = "bestModel", save_dir: str = "env_model") -> Tuple[list, list, list]:
    """
    train environment model based on sampled trajectories

    :param model: environment model
    :param features_train: features for training
    :param labels_train: labels for training
    :param x: x-coordinates of grid
    :param y: y-coordinates of grid
    :param re_no: Reynolds number
    :param epochs: number of epochs for training
    :param lr: learning rate
    :param batch_size: batch size
    :param save_model: option to save best model, default is True
    :param save_dir: path to directory where models should be saved
    :param save_name: name of the model saved, default is number of epoch
    :return: training and validation loss as list
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # optimizer settings
    optimizer = pt.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=(1.0e-4 / 1.0e-2) ** (1.0 / epochs))

    # lists for storing losses
    tot_train_loss, pred_loss, eq_loss, best_train_loss = [], [], [], 1e3

    # create dataset & dataloader -> dimensions always: [batch_size, N_features (or N_labels)]
    dataset_train = TensorDataset(features_train.requires_grad_(True), labels_train.requires_grad_(True))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        tot_loss_tmp, eq_loss_tmp, pred_loss_tmp = [], [], []

        # training loop
        for feature, label in dataloader_train:
            model.train()
            optimizer.zero_grad()

            # make prediction
            loss_pred = model.loss_prediction(feature=feature, len_x=x.size()[0], len_y=y.size()[0], u=label[:, :, 0],
                                              v=label[:, :, 1])
            loss_eq = model.loss_equation(feature, len_x=x.size()[0], len_y=y.size()[0], Re=re_no)
            loss_tot = loss_pred + loss_eq
            loss_tot.backward()
            optimizer.step()
            tot_loss_tmp.append(loss_tot.item())
            eq_loss_tmp.append(loss_eq.item())
            pred_loss_tmp.append(loss_pred.item())

        tot_train_loss.append(pt.mean(pt.tensor(tot_loss_tmp)))
        eq_loss.append(pt.mean(pt.tensor(eq_loss_tmp)))
        pred_loss.append(pt.mean(pt.tensor(pred_loss_tmp)))
        scheduler.step()

        # save best models
        if save_model:
            if tot_train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_dir}/{save_name}_train.pt")
                best_train_loss = tot_train_loss[-1]

        # print some info after every 100 epochs
        if epoch % 100 == 0:
            print(f"finished epoch {epoch}:\ttraining loss = {round(tot_train_loss[-1].item(), 8)}, \t"
                  f"equation loss = {round(eq_loss[-1].item(), 8)}, "
                  f"\tprediction loss = {round(pred_loss[-1].item(), 8)}")

    return tot_train_loss, eq_loss, pred_loss


def read_data(file: str, u_infty: int = 1, d: float = 0.1, mu: float = 1e-3, scale: bool = True) -> Tuple[dict, dict]:
    """
    reads in the flow field data from CFD

    :param file: path to the file containing the flow field data and the file name, it is assumed that the data was
                 created using the "export_cylinder2D_flow_field" function of this script
    :param u_infty: free stream velocity at inlet
    :param d: diameter of cylinder
    :param mu: kinematic viscosity
    :param scale: flag if loaded data should be scaled to [0, 1]
    :return: loaded data
    """
    # read flow field data and non-dimensionalize it
    data_mat = pickle.load(open(file, "rb"))
    U_star = data_mat["U"] / u_infty
    X_star = data_mat["xy_coord"] / d

    # [1] = [s] * [m/s] * [1/m] = t * u_infty * 1/d = t_star
    T_star = data_mat["t"] * u_infty * (1/d)

    # [1] = ([Pa] * [m]) / ([Pa * s] *[m / s]) = p_star
    P_star = data_mat["p"] * d / (mu * u_infty)

    # scale all available CFD data to interval [0, 1]
    if scale:
        V_star, min_max_v = scale_data(U_star[:, 1, :])
        U_star, min_max_u = scale_data(U_star[:, 0, :])
        Y_star, min_max_y = scale_data(X_star[:, 1].unsqueeze(-1))
        X_star, min_max_x = scale_data(X_star[:, 0].unsqueeze(-1))
        T_star, min_max_t = scale_data(T_star)
        P_star, min_max_p = scale_data(P_star)

        # feature matrix contains min- / max-values as well, but for all loaded data points
        min_max_vals = {"u": min_max_u, "v": min_max_v, "x": min_max_x, "y": min_max_y, "p": min_max_p, "t": min_max_t}

        # sanity check -> if not enough data points, then min == max and scaling returns 'nan'
        for key in min_max_vals:
            if abs(min_max_vals[key][0] - min_max_vals[key][1]).item() <= 10e-6:
                print("not enough values for normalization, increase 'ratio'!")
                exit()
    else:
        V_star = U_star[:, 1, :]
        U_star = U_star[:, 0, :]
        Y_star = X_star[:, 1].unsqueeze(-1)
        X_star = X_star[:, 0].unsqueeze(-1)
        min_max_vals = {}

    return {"x": X_star, "y": Y_star, "t": T_star, "u": U_star, "v": V_star, "p": P_star}, min_max_vals


def create_feature_label_pairs(data: dict) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    creates feature-label pairs for all available CFD data

    :param data: loaded CFD data
    :return: features and labels as tensors
    """
    # features = [x, y, t_i], labels = [u(t_i), v(t_i), p(t_i], assuming that the mesh is time-independent
    # the labels are NOT(!) the network output (output is Psi & p), but used for loss calculation with NS-eq.

    # feature shape: [N_time_steps, N_grid_points(x, y) + t_i], label shape: [N_time_steps, N_points, N_params(u, v, p)]
    shape_feature = (data["t"].size()[0], data["x"].size()[0] + data["y"].size()[0] + 1)
    shape_label = (data["t"].size()[0], data["u"].size()[0], 3)
    feature, label = pt.zeros(shape_feature), pt.zeros(shape_label)

    for idx, t in enumerate(data["t"]):
        feature[idx, :] = pt.concat([data["x"], data["y"], t.unsqueeze(-1)], dim=0).squeeze()
        label[idx, :, :] = pt.cat([data["u"][:, idx].unsqueeze(-1), data["v"][:, idx].unsqueeze(-1),
                                   data["p"][:, idx].unsqueeze(-1)], dim=1)

    return feature, label


def plot_losses(savepath: str, loss: Tuple[list, list, list], case: str = "NS_eq") -> None:
    """
    plot training- and equation- and prediction losses

    :param savepath: path where the plot should be saved
    :param loss: tensor containing all the losses
    :param case: append to save name
    :return: None
    """
    # plot training- and validation losses
    plt.plot(range(len(loss[0])), loss[0], color="blue", label="total training loss")
    plt.plot(range(len(loss[1])), loss[1], color="green", label="equation loss")
    plt.plot(range(len(loss[2])), loss[2], color="red", label="prediction loss")
    plt.xlabel("$epoch$ $number$", usetex=True, fontsize=14)
    plt.ylabel("$MSE$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncols=2)
    plt.yscale("log")
    plt.savefig("".join([savepath, f"/plots/losses_{case}.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    # Setup
    setup = {
        "path_data": r"./",  # path to file containing the flow field data
        "name_data": r"../cylinder2D_quasi_steady_flow_field.pkl",  # name of the file containing the flow field data
        "save_path": r"../TEST/",  # path for saving all results
        "Re": 100,  # Reynolds number
        "U_infty": 1,  # free stream velocity at inlet
        "epochs": 1000,  # number of epochs for training
    }

    # ensure reproducibility
    manual_seed(0)

    # load flow field data
    cfd_data, min_max_values = read_data(setup["path_data"] + setup["name_data"], d=0.1, u_infty=setup["U_infty"])

    # create feature-label pairs, mesh is the same for all time steps, then free up some memory
    features, labels = create_feature_label_pairs(cfd_data)

    # initialize models: input = grid + current time step, output = fields of u & psi
    pinn = FCModel(n_inputs=features.size()[-1], n_outputs=labels.size()[1] * 2)

    # train models
    losses = train_model(pinn, features, labels, epochs=setup["epochs"], x=cfd_data["x"], y=cfd_data["y"],
                         re_no=setup["Re"])

    # create directory for plots
    if not exists(setup["save_path"] + "plots"):
        os.mkdir(setup["save_path"] + "plots")

    # plot losses of model training
    plot_losses(setup["save_path"], losses)
    exit()

    # plot results
    print("\nstart predicting flow field...")
    compare_flow_fields(path=setup["save_path"], file_name=setup["name_data"], min_max=min_max_values)

    # make gifs from flow fields
    print("\ncreating gifs of flow field...")
    make_flow_gif(setup["save_path"], name="u", fps_num=10)
    make_flow_gif(setup["save_path"], name="v", fps_num=10)
    make_flow_gif(setup["save_path"], name="p", fps_num=10)
    print("Done.")
