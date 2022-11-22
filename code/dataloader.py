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
from typing import Union, Tuple
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
    t = [time for time in loader.write_times if float(time) > 7.0]

    # only read in [u, v] of U-field and discard the w-component, since flow is 2D, also ignore major parts of the wake
    # for testing purposes: discard cylinder, since mesh is very fine there
    mask = mask_box(loader.vertices[:, :2], lower=[0.5, -1], upper=[0.91, 1])
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

    # flow field need to be in order [N_points, N_dimensions, N_time_steps]
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


def read_data(file: str, portion: Union[float, int], u_infty: int = 1, d: float = 0.1, mu: float = 1e-3,
              scale: bool = True):
    """
    reads in the flow field data from CFD

    :param file: path to the file containing the flow field data and the file name, it is assumed that the data was
                 created using the "export_cylinder2D_flow_field" function of this script
    :param portion: ratio of the data for PINN training wrt to total amount of data
    :param u_infty: free stream velocity at inlet
    :param d: diameter of cylinder
    :param mu: kinematic viscosity
    :param scale: flag if loaded data should be scaled to [0, 1]
    :return:
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
        Y_star = X_star[:, 1]
        X_star = X_star[:, 0]
        min_max_vals = {}

    # Read the number of coordinate points N and the number of time steps T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # Turn the data into UU = (N_x_points * N_time_steps), VV = (N_y_points * N_time_steps), ...
    XX = np.tile(X_star, (1, T))
    YY = np.tile(Y_star, (1, T))
    TT = np.tile(T_star, (1, N)).T

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = U_star.flatten()[:, None]
    v = V_star.flatten()[:, None]
    p = P_star.flatten()[:, None]

    # shape = N_dimensions * N_parameters (parameters = x, y, t, u, v, p)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique) - 1, int(len(x_unique) * portion)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique) - 1, int(len(y_unique) * portion)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)

    # reconstruct the flow fields for the given (N*1D) data to 2D-mesh for each parameter
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

    return x, y, t, u, v, p, min_max_vals


def generate_eqp_rect(low_bound: Union[pt.Tensor, np.ndarray], up_bound: Union[pt.Tensor, np.ndarray], dimension: int,
                      points: int) -> pt.Tensor:
    """
    creates mesh for given number of points and dimensions for solving the NS-equations

    :param low_bound: min. value for each dimension (including temporal dimension)
    :param up_bound: max. value for each dimension (including temporal dimension)
    :param dimension: number of spacial and temporal dimensions
    :param points: number of grid points for solving the NS-equations
    :return: mesh with grid points for solving the NS-equations
    """
    # Generate rectangular domain equation points
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    per = np.random.permutation(eqa_xyzt.shape[0])
    new_xyzt = eqa_xyzt[per, :]
    eqa_points = pt.from_numpy(new_xyzt).float()
    return eqa_points


def prepare_data(x: pt.Tensor, y: pt.Tensor, t: pt.Tensor, u: pt.Tensor, v: pt.Tensor, p: pt.Tensor,
                 n_eq_points: int = 10000, dimensions: int = 3, batch_ratio: float = 0.005) -> dict:
    """
    TODO

    :param x: x-coordinates
    :param y: y-coordinates
    :param t: all available time steps
    :param u: data of velocity field for u (for all available time steps)
    :param v: data of velocity field for v (for all available time steps)
    :param p: data of pressure field (for all available time steps)
    :param n_eq_points: N grid points for solving NS-equations
    :param dimensions: spacial + temporal dimensions
    :param batch_ratio: TODO
    :return: TODO
    """
    x_random = shuffle_data(x, y, t, u, v, p)

    # since data is normalized to [0, 1], the bounds for each parameter within parameter space is [0, 1]
    points_eq = generate_eqp_rect(np.zeros(3), np.ones(3), dimensions, n_eq_points)
    batch_size_data = int(batch_ratio * x_random.shape[0])

    if batch_size_data == 0:
        print("not enough data points, increase the 'ratio' parameter!")
        exit()
    return {"points_eq": points_eq, "batch_size_data": batch_size_data, "x_random": x_random,
            "eq_iter": int(points_eq.size()[0] / int(batch_ratio * points_eq.shape[0])),
            "batch_size_eq": int(batch_ratio*points_eq.shape[0]), "inner_iter": int(x_random.size()[0]/batch_size_data)}


def shuffle_data(x, y, t, u, v, p):
    """
    randomly shuffle the data, so that it is not ordered wrt to time series

    :param x: data of x-dimension
    :param y: data of x-dimension
    :param t: data of temporal dimension (all available time steps)
    :param u: velocity field in x-direction of all time steps
    :param v: velocity field in y-direction of all time steps
    :param p: pressure field of all time steps
    :return: tensor containing the shuffled data
    """
    x_total = pt.cat([x, y, t, u, v, p], 1)
    x_total_arr = x_total.data.numpy()
    np.random.shuffle(x_total_arr)
    return pt.tensor(x_total_arr)


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
    # export flow field data of cylinder2D case for PINN training
    path = r"/home/janis/Hiwi_ISM/py_scripts_exercises/exercise_9_pinn_cylinder2d/"
    export_cylinder2D_flow_field(path + "uncontrolled/", path)
