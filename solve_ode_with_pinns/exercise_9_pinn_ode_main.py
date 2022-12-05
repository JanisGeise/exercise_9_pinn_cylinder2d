"""
    solve ODE's using PINN's

    1. test case ('exponential_decay_const_k.py'):
        dx/dt = -kx     with    x(t=0) = 1, k = const.

    2. test case ('exponential_decay_variable_k.py'):
        dx/dt = -kx     with    x(t=0) = 1, k != const.

"""
import torch as pt
from os import path, mkdir

from exponential_decay_const_k import wrapper_execute_training


if __name__ == "__main__":
    # setup
    setup = {
        "load_path": r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/",     # path to target directory (top-level)
        "n_epochs": 500,                        # number of epochs for model training
        "k_const": 0.25,                        # k-factor for ODE with k = const.
        "k_variable": pt.linspace(0.2, 1, 5)    # k-factors for ODE with k != const.
    }

    # create directory for plots
    if not path.exists("".join([setup["load_path"], "plots"])):
        mkdir("".join([setup["load_path"], "plots"]))

    # ensure reproducibility
    pt.manual_seed(0)

    # first ODE, k = const.
    wrapper_execute_training(load_path=setup["load_path"], k=setup["k_const"], n_epochs=setup["n_epochs"])
