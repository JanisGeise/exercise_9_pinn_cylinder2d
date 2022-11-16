"""
    post-processing script

    Note: this code is a modified version of the code presented in this repository:
          https://github.com/Shengfeng233/PINN-for-NS-equation
"""
import pickle
import imageio
import numpy as np
import torch as pt
from matplotlib import colors
from pandas import read_csv
from typing import Union, Tuple
import matplotlib.pyplot as plt

from dataloader import read_data
from train_models import PINN
from train_models import f_equation_inverse


def plot_loss(path: str, name_loss_file: str = "loss") -> None:
    """
    plots losses of PINN training

    :param path: path to the file containing the losses
    :param name_loss_file: name of the file containing the losses
    :return: None
    """

    # import loss data
    file_loss = read_csv("".join([path, name_loss_file, ".csv"]), header=None)
    loss = file_loss.values
    n_epochs = range(len(loss[:, 0]))

    # plot losses
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(n_epochs, loss[:, 0], marker="None", color="blue", label="total loss")
    ax.semilogy(n_epochs, loss[:, 1], marker="None", color="green", label="prediction loss")
    ax.semilogy(n_epochs, loss[:, 2], marker="None", color="red", label="equation loss")
    ax.set_title("$MSE$ $loss$ $of$ $training$", usetex=True, fontsize=16)
    ax.set_xlabel("$epoch$ $number$", usetex=True, fontsize=14)
    ax.set_ylabel("$MSE$ $loss$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncol=3)
    fig.tight_layout()
    plt.savefig(path + "/plots/loss.png")
    plt.close("all")


def compare_at_select_time_series(path: str, file_name: str, lower_time: Union[float, int] = 4.0,
                                  upper_time: Union[float, int] = 8.0, d: float = 0.1, mu: float = 1e-3,
                                  u_infty: Union[int, float] = 1, model_name: str = "model_train") -> None:
    """
    predicts flow fields based on a trained model and plots them compared to the real flow fields

    :param path: save path for plots
    :param file_name: name of the file containing the real flow field data
    :param lower_time: starting time for prediction
    :param upper_time: end time for prediction
    :param model_name: name of the trained model
    :param d: diameter of cylinder
    :param mu: kinematic viscosity
    :param u_infty: free stream velocity at inlet
    :return: None
    """
    # read in the data and model, TODO: everything here is still quite inefficient...
    xy_cfd = pickle.load(open(file_name, "rb"))["xy_coord"]
    uv_cfd = pickle.load(open(file_name, "rb"))["U"]
    p_cfd = pickle.load(open(file_name, "rb"))["p"]

    x, y, t, u, v, p, _ = read_data(file_name, 1)
    data_stack = np.concatenate((x, y, t, u, v, p), axis=1)
    pinn_net = PINN()
    pinn_net.load_state_dict(pt.load("".join([path, model_name, ".pt"]), map_location="cpu"))
    pinn_net.eval()

    # min-max scaling for plots
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
    del data_stack, u, v, p

    # Keep the coordinates that are not repeated in the data set
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(x).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x, y)

    # determine the available time steps and create array with them (1st: reverse non-dimensionalization)
    dt = round((t[1]*(d * u_infty)).item() - (t[0]*(d * u_infty)).item(), 6)
    n = int((upper_time - lower_time) / dt) + 1
    time_lists = np.linspace(lower_time, upper_time, n)

    for idx, select_time in enumerate(time_lists):
        print(f"\tpredicting time step t = {round(select_time, 2)}s")
        select_time = round(float(select_time), 6)
        x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
        t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

        x_selected = pt.tensor(x_flatten, requires_grad=True, dtype=pt.float32)
        y_selected = pt.tensor(np.ndarray.flatten(mesh_y).reshape(-1, 1), requires_grad=True, dtype=pt.float32)
        t_selected = pt.tensor(t_flatten, requires_grad=True, dtype=pt.float32)

        # predict the flow field of the current time step
        u_predict, v_predict, p_predict = f_equation_inverse(x_selected, y_selected, t_selected, pinn_net.eval())
        u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
        v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
        p_predict = p_predict.data.numpy().reshape(mesh_x.shape)

        # plot the velocity- and pressure field for each given time step
        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], uv_cfd[:, 0, idx]*u_infty], u_predict*u_infty,
                                   select_time, name="u", min_value=min_data[0, 3], max_value=max_data[0, 3])
        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], uv_cfd[:, 1, idx]*u_infty], v_predict*u_infty,
                                   select_time, name="v", min_value=min_data[0, 4], max_value=max_data[0, 4])
        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], p_cfd[:, idx] * (mu * u_infty) / d],
                                   p_predict * (mu * u_infty) / d, select_time, name="p", min_value=min_data[0, 5],
                                   max_value=max_data[0, 5])

        # free up some memory
        del u_predict, v_predict, p_predict, x_selected, y_selected, t_selected, x_flatten, t_flatten


def plot_comparison_flow_field(path: str, q_selected, q_predict, select_time, min_value, max_value, name="u") -> None:
    """

    :param path: save path for plots
    :param q_selected: CFD field data
    :param q_predict: corresponding predicted field data by the PINN's
    :param select_time: time frame of flow field
    :param min_value: min. value within flow field for color bar limits
    :param max_value: max. value within flow field for color bar limits
    :param name: name of the flow filed, e.g. u, v or p
    :return: None
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    v_norm = colors.Normalize(vmin=min_value, vmax=max_value)
    for i in range(2):
        if i == 0:
            ax[i].set_title(f"$Real$ $flow$ $field$ $at$ $t = " + "{:.2f}s$".format(select_time), usetex=True)
            # TODO: norm doesn't have any effect, bounds of color bar changing every time step...
            real = ax[i].tricontourf(q_selected[0], q_selected[1], q_selected[2], levels=200, cmap="jet", norm=v_norm)
            cb = fig.colorbar(real, ax=ax[i], format="{x:.2f}", norm=v_norm)
        else:
            ax[i].set_title(f"$Predicted$ $flow$ $field$ $at$ $t = " + "{:.2f}s$".format(select_time), usetex=True)
            # TODO: norm doesn't have any effect, bounds of color bar changing every time step...
            pred = ax[i].contourf(q_predict, levels=200, cmap="jet", norm=v_norm)
            cb = fig.colorbar(pred, ax=ax[i], format="{x:.2f}", norm=v_norm)
            # TODO: this gives a warning from plt
            ax[i].set_yticklabels(["{:.2f}".format(t) for t in ax[0].get_yticks()])
            ax[i].set_xticklabels(["{:.2f}".format(t) for t in ax[0].get_xticks()])
        ax[i].set_ylabel("$y-coordinate$", usetex=True, fontsize=14)

        if name != "p":
            cb.set_label(f"${name}$ $\quad[m/s]$", usetex=True, labelpad=20, fontsize=14)
        else:
            cb.set_label(f"${name}$ $\quad[Pa]$", usetex=True, labelpad=20, fontsize=14)

    ax[0].set_xticklabels([])
    ax[1].set_xlabel("$x-coordinate$", usetex=True, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    plt.savefig("".join([path, "/plots/time", "{:.2f}".format(select_time), name, ".png"]))
    plt.close("all")


def make_flow_gif(path, lower_time, upper_time, interval=0.01, name='q', fps_num=5, save_name: str = "test") -> None:
    """

    :param path: save path
    :param lower_time: starting time for gif
    :param upper_time: end time for gif
    :param interval: dt between two time frames
    :param name: name of the fields used for creating gif
    :param fps_num: fps
    :param save_name: save name of gif
    :return: None
    """
    gif_images = []
    n = int((upper_time - lower_time) / interval) + 1
    time_lists = np.linspace(lower_time, upper_time, n)
    for t in time_lists:
        gif_images.append(imageio.v2.imread("".join([path, "/plots/", "time", "{:.2f}".format(round(float(t), 3)), name,
                                                     ".png"])))
    imageio.mimsave("".join([path, "/plots/1_data_", name, save_name, ".gif"]), gif_images, fps=fps_num)


if __name__ == "__main__":
    pass
