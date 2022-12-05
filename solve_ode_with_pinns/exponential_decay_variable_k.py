"""
    2. case: solve ODE's using PINN's, here the implementation for:

        dx/dt = -kx     with    x(t=0) = 1, k != const.

"""

from utils import *
from exponential_decay_const_k import compute_analytical_solution


class PinnVariableK(PINN):
    def compute_loss_equation(self, model, t: pt.Tensor, k: pt.Tensor) -> pt.Tensor:
        # in contrast to k = const., here k is input to model
        pass

    def compute_loss_prediction(self, *args):
        pass


if __name__ == "__main__":
    pass
