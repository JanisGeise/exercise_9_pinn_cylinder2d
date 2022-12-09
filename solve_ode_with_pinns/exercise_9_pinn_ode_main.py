"""
    solve ODE's using PINN's

    1. test case: exponential decay with const. decay factor ('exponential_decay_const_k.py'):

        .. math:: \\frac{dx}{dt} = -kx

            x(t=0) = 1

            k = const.

    2. test case: exponential decay with variable decay factor ('exponential_decay_variable_k.py'):

        .. math:: \\frac{dx}{dt} = -kx

            x(t=0) = 1

            k \\ne const.

    3. test case: 1D-convection equation with const. heat coefficient ('convection_equation_1D.py'):

        .. math:: \\frac{\partial u}{\partial t} = c * \\frac{\partial^2 u}{\partial x^2}

        with:
            c = const.

            IC: u(t = 0, x) = 0

            BC: u(t > 0, x = 0) = 1

"""
from os import path, mkdir
from torch import manual_seed, linspace

from exponential_decay_const_k import wrapper_execute_training as first_ode
from exponential_decay_variable_k import wrapper_execute_training as second_ode


if __name__ == "__main__":
    # setup
    setup = {
        "load_path": r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/",     # path to target directory (top-level)
        "n_epochs": 2000,                        # number of epochs for model training (should be >= 500, better 1e3)
        "k_const": 0.55,                         # k-factor for ODE with k = const.
        "k_variable": linspace(0.05, 0.95, 6),       # k-factors for ODE with k != const.
    }

    # create directory for plots
    if not path.exists("".join([setup["load_path"], "plots"])):
        mkdir("".join([setup["load_path"], "plots"]))

    # ensure reproducibility
    manual_seed(0)

    # first ODE, k = const.
    first_ode(load_path=setup["load_path"], k=setup["k_const"], n_epochs=setup["n_epochs"])

    # 2nd ODE, k != const.
    second_ode(load_path=setup["load_path"], k=setup["k_variable"], n_epochs=setup["n_epochs"])

    # 3rd ODE
