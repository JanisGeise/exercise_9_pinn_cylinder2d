"""
    main script for PINN training

    Note: this code is a modified version of the code presented in this repository:
          https://github.com/Shengfeng233/PINN-for-NS-equation
"""
from os import mkdir
from os.path import exists
from torch import manual_seed
from numpy import random

from train_models import train_pinns
from post_process_results import plot_loss, compare_flow_fields, make_flow_gif
from dataloader import read_data, prepare_data


if __name__ == "__main__":
    # Setup
    setup = {
        "path_data": r"./",                                         # path to file containing the flow field data
        "name_data": r"../cylinder2D_quasi_steady_flow_field.pkl",  # name of the file containing the flow field data
        "save_path": r"../test_training_cylinder2D_1000_points_2nd/",         # path for saving all results
        "Re": 100,                                  # Reynolds number
        "U_infty": 1,                               # free stream velocity at inlet
        "epochs": 1000,                               # number of epochs for training
        "ratio": 1.0,            # ratio for creating sparse data (from total amount of data)
        "N_points": 1e3,          # N grid points in each dimension for the equations, mainly determines the runtime
        }

    # ensure reproducibility
    manual_seed(0)
    random.seed(0)

    # load flow field data
    x, y, t, u, v, p, min_max_vals = read_data(setup["path_data"] + setup["name_data"], setup["ratio"], d=0.1,
                                               u_infty=setup["U_infty"])

    # use LHS to generate sparse data points from loaded CFD data and free up some space
    data = prepare_data(x, y, t, u, v, p)
    del x, y, t, u, v, p

    # When the number of iters does not match, the program will be automatically exit
    if data["inner_iter"] != data["eq_iter"]:
        print(data["inner_iter"], data["eq_iter"])
        print('Poor batch size, need to reassign the value of "ratio"')
        exit()

    # train models with the sparse data
    train_pinns(data["inner_iter"], data["x_random"], data["batch_size_data"], data["batch_size_eq"], data["points_eq"],
                re_no=setup["Re"], epochs=setup["epochs"], path=setup["save_path"])

    # create directory for plots
    if not exists(setup["save_path"] + "plots"):
        mkdir(setup["save_path"] + "plots")

    # plot losses of model training
    plot_loss(setup["save_path"])

    # plot results
    print("\nstart predicting flow field...")
    compare_flow_fields(path=setup["save_path"], file_name=setup["name_data"], min_max=min_max_vals)

    # make gifs from flow fields
    print("\ncreating gifs of flow field...")
    make_flow_gif(setup["save_path"], name="u", fps_num=10)
    make_flow_gif(setup["save_path"], name="v", fps_num=10)
    make_flow_gif(setup["save_path"], name="p", fps_num=10)
    print("Done.")
