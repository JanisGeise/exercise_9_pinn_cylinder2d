"""
    2. case: solve ODE's using PINN's, here the implementation for:

        .. math:: \\frac{\partial u}{\partial t} = c * \\frac{\partial^2 u}{\partial x^2}

        with:
            c = const.

            IC: u(t = 0, x) = 0

            BC: u(t > 0, x = 0) = 1

"""

from utils import PINN, train_pinn


class PinnHeatEquation(PINN):
    def compute_loss_equation(self, *args):
        pass

    def compute_loss_prediction(self, *args):
        pass


def compute_analytical_solution():
    pass


if __name__ == "__main__":
    pass
