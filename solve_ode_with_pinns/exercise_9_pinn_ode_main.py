"""
    solve ODE's using PINN's

    1. test case: exponential decay with const. decay factor ('exponential_decay_const_k.py'):
        dx/dt = -kx     with    x(t=0) = 1, k = const.

    2. test case: exponential decay with variable decay factor ('exponential_decay_variable_k.py'):
        dx/dt = -kx     with    x(t=0) = 1, k != const.

    3. test case: 1D-heat equation with const. heat coefficient ('heat_equation_1D.py'):
        du / dt = c * d²u / dx²

        with:
            IC: u(t = 0, x) = 0
            BC: u(t > 0, x = 0) = 1
                c = const.

"""
import torch as pt
from os import path, mkdir

from exponential_decay_const_k import wrapper_execute_training as first_ode
from exponential_decay_variable_k import wrapper_execute_training as second_ode


if __name__ == "__main__":
    # setup
    setup = {
        "load_path": r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/",     # path to target directory (top-level)
        "n_epochs": 2000,                        # number of epochs for model training (should be >= 500, better 1e3)
        "k_const": 0.55,                        # k-factor for ODE with k = const.
        "k_variable": pt.linspace(0.05, 1, 6)    # k-factors for ODE with k != const.
    }

    # create directory for plots
    if not path.exists("".join([setup["load_path"], "plots"])):
        mkdir("".join([setup["load_path"], "plots"]))

    # ensure reproducibility
    pt.manual_seed(0)

    # first ODE, k = const.
    first_ode(load_path=setup["load_path"], k=setup["k_const"], n_epochs=setup["n_epochs"])

    # 2nd ODE, k != const.
    second_ode(load_path=setup["load_path"], k=setup["k_variable"], n_epochs=setup["n_epochs"])

    # 3rd ODE
