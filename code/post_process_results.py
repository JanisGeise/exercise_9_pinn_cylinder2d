"""
    post-processing script

    Note: this code is a modified version of the code presented in this repository:
          https://github.com/Shengfeng233/PINN-for-NS-equation
"""
import pickle
import imageio
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from typing import Union
from pandas import read_csv
from natsort import natsorted
from matplotlib.patches import Circle

from dataloader import read_data, rescale_data
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


def compare_flow_fields(path: str, file_name: str, min_max: dict, d: float = 0.1, mu: float = 1e-3,
                        u_infty: Union[int, float] = 1, model_name: str = "model_train") -> None:
    """
    predicts flow fields based on a trained model and plots them compared to the real flow fields

    :param path: save path for plots
    :param file_name: name of the file containing the real flow field data
    :param min_max: min- / max-values for scaling the loaded data
    :param model_name: name of the trained model
    :param d: diameter of cylinder
    :param mu: kinematic viscosity
    :param u_infty: free stream velocity at inlet
    :return: None
    """
    # read in the data
    xy_cfd = pickle.load(open(file_name, "rb"))["xy_coord"]
    uv_cfd = pickle.load(open(file_name, "rb"))["U"]
    p_cfd = pickle.load(open(file_name, "rb"))["p"]
    x, y, t, u, v, p, _ = read_data(file_name, 1, scale=True)

    # read in the best model
    pinn_net = PINN()
    pinn_net.load_state_dict(pt.load("".join([path, model_name, ".pt"]), map_location="cpu"))
    pinn_net.eval()

    # Keep the coordinates that are not repeated in the data set
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(x).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x, y)

    for idx, select_time in enumerate(np.unique(t)):
        # rescale time and reverse non-dimensionalization
        real_time = (rescale_data(pt.tensor(select_time), min_max["t"]) * (d / u_infty)).item()
        print(f"\tpredicting time step t = {round(real_time, 2)}s")

        select_time = round(float(select_time), 6)
        x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
        t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

        x_selected = pt.tensor(x_flatten, requires_grad=True, dtype=pt.float32)
        y_selected = pt.tensor(np.ndarray.flatten(mesh_y).reshape(-1, 1), requires_grad=True, dtype=pt.float32)
        t_selected = pt.tensor(t_flatten, requires_grad=True, dtype=pt.float32)
        del x_flatten, t_flatten

        # predict the flow field of the current time step
        u_predict, v_predict, p_predict = f_equation_inverse(x_selected, y_selected, t_selected, pinn_net.eval())

        # reshape & re-scale the flow- and pressure fields
        u_predict = (rescale_data(u_predict.detach(), min_max["u"])).flatten()
        v_predict = (rescale_data(v_predict.detach(), min_max["v"])).flatten()
        p_predict = (rescale_data(p_predict.detach(), min_max["p"])).flatten()
        x_selected = (rescale_data(x_selected.detach(), min_max["x"])).flatten()
        y_selected = (rescale_data(y_selected.detach(), min_max["y"])).flatten()

        # plot the flow field for each time step, for CFD, the x- & y-coord. are already in [m] (no "*d" necessary)
        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], uv_cfd[:, 0, idx]*u_infty],
                                   [x_selected*d, y_selected*d, u_predict*u_infty],
                                   real_time, name="u", min_value=min_max["u"][0], max_value=min_max["u"][1])

        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], uv_cfd[:, 1, idx]*u_infty],
                                   [x_selected * d, y_selected * d, v_predict * u_infty], real_time, name="v",
                                   min_value=min_max["v"][0], max_value=min_max["v"][1])

        plot_comparison_flow_field(path, [xy_cfd[:, 0], xy_cfd[:, 1], p_cfd[:, idx] * (mu * u_infty) / d],
                                   [x_selected * d, y_selected * d, p_predict * (mu * u_infty) / d], real_time,
                                   name="p", min_value=min_max["p"][0], max_value=min_max["p"][1])

        # free up some memory
        del u_predict, v_predict, p_predict, x_selected, y_selected, t_selected


def plot_comparison_flow_field(path: str, q_selected, q_predict, select_time, min_value, max_value, name="") -> None:
    """
    plot the real flow field from CFD in comparison to the predicted one

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

    # dummy point for color bar
    x, y = (q_selected[0][0], q_selected[0][0]), (q_selected[1][0], q_selected[1][0])
    c = (pt.min(q_selected[2]).item(), pt.max(q_selected[2]).item())
    dummy = plt.scatter(x, y, s=0.1, cmap="jet", c=c, vmin=round(min_value.item(), 1), vmax=round(max_value.item(), 1))

    # bounds for axis limits
    x_min, x_max = round(pt.min(q_selected[0]).item(), 1), round(pt.max(q_selected[0]).item(), 1)
    y_min, y_max = round(pt.min(q_selected[1]).item(), 1), round(pt.max(q_selected[1]).item(), 1)

    for i in range(2):
        if i == 0:
            ax[i].set_title(f"$Real$ $flow$ $field$ $at$ $t = " + "{:.2f}s$".format(select_time), usetex=True)
            if name == "p":
                ax[i].tricontourf(q_selected[0], q_selected[1], q_selected[2], levels=200, cmap="jet")
            else:
                ax[i].tricontourf(q_selected[0], q_selected[1], q_selected[2], levels=200, cmap="jet",
                                  vmin=round(min_value.item(), 1), vmax=round(max_value.item(), 1))
            cb = fig.colorbar(dummy, ax=ax[i], format="{x:.2f}")
            ax[i].set_xticks(np.arange(x_min, x_max+0.1, 0.1))
            ax[i].set_yticks(np.arange(y_min, y_max+0.05, 0.05))
            # ax[i].add_patch(Circle((0.2, 0.2), radius=0.05, edgecolor="black", facecolor="white", linewidth=2))
        else:
            ax[i].set_title(f"$Predicted$ $flow$ $field$ $at$ $t = " + "{:.2f}s$".format(select_time), usetex=True)
            if name == "p":
                ax[i].tricontourf(q_predict[0], q_predict[1], q_predict[2], levels=200, cmap="jet")
            else:
                ax[i].tricontourf(q_predict[0], q_predict[1], q_predict[2], levels=200, cmap="jet",
                                  vmin=round(min_value.item(), 1), vmax=round(max_value.item(), 1))

            cb = fig.colorbar(dummy, ax=ax[i], format="{x:.2f}")
            # ax[i].add_patch(Circle((0.2, 0.2), radius=0.05, edgecolor="black", facecolor="white", linewidth=2))
            ax[i].set_xticks(np.arange(x_min, x_max+0.1, 0.1))
            ax[i].set_yticks(np.arange(y_min, y_max+0.05, 0.05))
        ax[i].set_ylabel("$y\quad[m]$", usetex=True, fontsize=14)

        if name != "p":
            cb.set_label(f"${name}$ $\quad[m/s]$", usetex=True, labelpad=20, fontsize=14)
        else:
            cb.set_label(f"${name}$ $\quad[Pa]$", usetex=True, labelpad=20, fontsize=14)

    ax[0].set_xticklabels([])
    ax[1].set_xlabel("$x\quad[m]$", usetex=True, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    plt.savefig("".join([path, "/plots/time", "{:.2f}".format(select_time), name, ".png"]))
    plt.close("all")


def make_flow_gif(path, name="u", fps_num=5, save_name: str = "flow_field_") -> None:
    """
    create gif from of flow field plots for all predicted time steps

    :param path: save path
    :param name: name of the fields used for creating gif
    :param fps_num: fps
    :param save_name: save name of gif
    :return: None
    """
    img = [imageio.v2.imread(img) for img in natsorted(glob("".join([path, "/plots/", "time*", name, ".png"])))]
    imageio.mimsave("".join([path, "/plots/", save_name, name, ".gif"]), img, fps=fps_num)


if __name__ == "__main__":
    pass
