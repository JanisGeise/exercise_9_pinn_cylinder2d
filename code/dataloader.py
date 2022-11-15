"""
    this script handles all the data IO & pre-processing

    Note: this code is a modified version of the code presented in this repository:
          https://github.com/Shengfeng233/PINN-for-NS-equation
"""
import sys
import pickle

import numpy as np
import torch as pt
from pyDOE import lhs
from typing import Union
import matplotlib.pyplot as plt

sys.path.insert(0, "/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/bin")
sys.path.insert(0, "/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/lib/python3.8/site-packages")

from flowtorch.data import FOAMDataloader, mask_box


def export_cylinder2D_flow_field(load_path: str, save_path: str) -> None:
    """
    extracts quasi-steady flow of cylinder2D case and exports it for PINN training

    :param load_path: path to cylinder2D case run in openfoam
    :param save_path: path where to save the flow field data
    :return: None
    """
    # load the snapshots for t < 6...8 s, because N_time_steps need to be dividable by ratio for batch size
    loader = FOAMDataloader(load_path)
    t = [time for time in loader.write_times if float(time) > 6.0]

    # only read in [u, v] of U-field and discard the w-component, since flow is 2D, also ignore major parts of the wake
    # for testing purposes: discard cylinder, since mesh is very fine there
    mask = mask_box(loader.vertices[:, :2], lower=[0.35, -1], upper=[0.75, 1])
    u_field = pt.zeros((mask.sum().item(), len(t)), dtype=pt.float32)
    v_field = pt.zeros((mask.sum().item(), len(t)), dtype=pt.float32)
    p_field = pt.zeros((mask.sum().item(), len(t)), dtype=pt.float32)
    for i, time in enumerate(t):
        u_field[:, i] = pt.masked_select(loader.load_snapshot("U", time)[:, 0], mask)
        v_field[:, i] = pt.masked_select(loader.load_snapshot("U", time)[:, 1], mask)
        p_field[:, i] = pt.masked_select(loader.load_snapshot("p", time), mask)

    # apply mask in order to reduce amount of data
    x = pt.masked_select(loader.vertices[:, 0], mask)
    y = pt.masked_select(loader.vertices[:, 1], mask)

    # flow field need to be in order [N_points, N_time_steps, N_dimensions]
    uv = pt.zeros((u_field.size()[0], 2, len(t)))
    for i in range(len(t)):
        uv[:, 0, i] = u_field[:, i]
        uv[:, 1, i] = v_field[:, i]

    data_out = {"t": pt.tensor(list(map(float, t))).reshape((u_field.size()[-1], 1)), "p": p_field,
                "xy_coord": pt.stack([x, y], dim=-1), "U": uv}

    """
    fig, ax = plt.subplots()
    ax.scatter(loader.vertices[:, 0], loader.vertices[:, 1], s=0.5, c=mask)
    ax.set_aspect("equal", 'box')
    ax.set_xlim(0.0, 2.2)
    ax.set_ylim(0.0, 0.41)
    fig.tight_layout()
    plt.show()
    """

    with open("".join([save_path, f"cylinder2D_quasi_steady_flow_field.pkl"]), "wb") as f:
        pickle.dump(data_out, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_data(file: str, portion: Union[float, int]):
    # Read flow field data
    data_mat = pickle.load(open(file, "rb"))
    U_star = data_mat["U"]
    X_star = data_mat["xy_coord"]
    T_star = data_mat["t"]
    P_star = data_mat["p"]

    # Read the number of coordinate points N and the number of time steps T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # TODO: make this more efficient if possible
    # Turn the data into x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((3, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique) - 1, int(len(x_unique) * portion)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique) - 1, int(len(y_unique) * portion)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)

    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)
    t = t[index_all].reshape(-1, 1)
    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = pt.tensor(x, dtype=pt.float32)
    y = pt.tensor(y, dtype=pt.float32)
    t = pt.tensor(t, dtype=pt.float32)
    feature_mat = pt.tensor(feature_mat, dtype=pt.float32)
    return x, y, t, u, v, p, feature_mat


def generate_eqp_rect(low_bound, up_bound, dimension, points):
    # Generate rectangular domain equation points
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    per = np.random.permutation(eqa_xyzt.shape[0])
    new_xyzt = eqa_xyzt[per, :]
    eqa_points = pt.from_numpy(new_xyzt).float()
    return eqa_points


def prepare_data(x, y, t, u, v, p, loaded_data, n_eq_points: int = 10000, dimensions: int = 3,
                 batch_ratio: float = 0.005):
    # TODO: throws division by zero error if batch_ratio < 2*10e-2
    x_random = shuffle_data(x, y, t, u, v, p)
    lb = np.array([loaded_data.data.numpy()[1, 0], loaded_data.data.numpy()[1, 1], loaded_data.data.numpy()[1, 2]])
    ub = np.array([loaded_data.data.numpy()[0, 0], loaded_data.data.numpy()[0, 1], loaded_data.data.numpy()[0, 2]])
    points_eq = generate_eqp_rect(lb, ub, dimensions, n_eq_points)
    batch_size_data = int(batch_ratio * x_random.shape[0])

    return {"points_eq": points_eq, "batch_size_data": batch_size_data, "x_random": x_random,
            "eq_iter": int(points_eq.size(0) / int(batch_ratio * points_eq.shape[0])),
            "batch_size_eq": int(batch_ratio * points_eq.shape[0]), "inner_iter": int(x_random.size(0) / batch_size_data)}


def shuffle_data(x, y, t, u, v, p):
    X_total = pt.cat([x, y, t, u, v, p], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = pt.tensor(X_total_arr)
    return X_total_random


if __name__ == "__main__":
    # export flow field data of cylinder2D case for PINN training
    path = r"/home/janis/Hiwi_ISM/py_scripts_exercises/pinn_cylinder2d/"
    export_cylinder2D_flow_field(path + "uncontrolled/", path)
