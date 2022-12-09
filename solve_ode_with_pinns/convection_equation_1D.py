"""
    2. case: solve ODE's using PINN's, here the implementation for:

        .. math:: \\frac{\partial u}{\partial t} = c * \\frac{\partial^2 u}{\partial x^2}

        with:
            c = const.

            IC: u(t = 0, x) = 0

            BC: u(t > 0, x = 0) = 1

"""
import matplotlib.pyplot as plt

from utils import *


class PinnHeatEquation(PINN):
    def compute_loss_equation(self, *args):
        pass

    def compute_loss_prediction(self, *args):
        pass


def compute_analytical_solution():
    pass


def plot_sampled_points():
    pass


def plot_prediction_vs_analytical_solution():
    pass


def wrapper_execute_training():
    pass


if __name__ == "__main__":
    pass
