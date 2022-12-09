"""
    2. case: solve ODE's using PINN's, here the implementation for:

        du / dt = c * d²u / dx²

        with:
            IC: u(t = 0, x) = 0
            BC: u(t > 0, x = 0) = 1
                c = const.

"""

from utils import PINN, train_pinn


class PinnHeatEquation(PINN):
    def compute_loss_equation(self, *args):
        pass

    def compute_loss_prediction(self, *args):
        pass


if __name__ == "__main__":
    pass
