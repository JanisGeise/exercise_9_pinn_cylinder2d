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

    3. test case: 1D-diffusion equation with const. diffusion coefficient ('diffusion_equation_1D.py'):

        .. math:: \\frac{\partial c}{\partial t} = \\alpha \\frac{\partial^2 c}{\partial x^2}

        with:
            alpha = const.

            IC: c(t = 0, x) = 0

            BC: c(t > 0, x = 0) = 1

"""
from os import path, mkdir
from torch import manual_seed, linspace

from exponential_decay_const_k import wrapper_execute_training as first_ode
from exponential_decay_variable_k import wrapper_execute_training as second_ode
from diffusion_equation_1D import wrapper_execute_training as first_pde


if __name__ == "__main__":
    # setup
    setup = {
        "load_path": r"/home/janis/Hiwi_ISM/ml-cfd-lecture/exercises/",     # path to target directory (top-level)
        "k_const": 0.55,                            # k-factor for ODE with k = const.
        "k_variable": linspace(0.05, 0.95, 6),      # k-factors for ODE with k != const.
        "x_min": 0,                                 # min. x-value for PDE (3. case)
        "x_max": 1,                                 # max. x-value for PDE (3. case)
        "t_min": 0.00,                              # min. t-value for PDE (3. case)
        "t_max": 100,                               # max. t-value for PDE (3. case)
        "alpha": 1e-3,                              # diffusion coefficient for PDE (3. case)
    }

    # create directory for plots
    if not path.exists("".join([setup["load_path"], "plots"])):
        mkdir("".join([setup["load_path"], "plots"]))

    # ensure reproducibility
    manual_seed(0)

    # first ODE, k = const.
    first_ode(load_path=setup["load_path"], k=setup["k_const"], n_epochs=500)

    # 2nd ODE, k != const.
    second_ode(load_path=setup["load_path"], k=setup["k_variable"], n_epochs=2000)

    # 1st PDE: diffusion equation
    first_pde(load_path=setup["load_path"], x_min=setup["x_min"], x_max=setup["x_max"], n_epochs=10000,
              alpha=setup["alpha"], t_start=setup["t_min"], t_end=setup["t_max"])
