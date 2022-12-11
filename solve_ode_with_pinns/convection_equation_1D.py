"""
    2. case: solve ODE's using PINN's, here the implementation for:

        .. math:: \\frac{\partial c}{\partial t} = \\alpha \\frac{\partial^2 c}{\partial x^2}

        with:
            alpha = const.

            IC: c(t = 0, x) = 0

            BC: c(t > 0, x = 0) = 1

"""
import matplotlib.pyplot as plt

from utils import *


class PinnHeatEquation(PINN):
    def compute_loss_equation(self, model, model_input: pt.Tensor, alpha: Union[int, float, pt.Tensor]) -> pt.Tensor:
        """
        compute the equation loss

        :param model: PINN model
        :param model_input: x- and t-values
        :param alpha: diffusion factor
        :return: equation loss
        """
        # make prediction: the variable 't' needs to be defined as input, otherwise error since it wouldn't be part of
        # the graph (if the name is not the same...) TODO
        t, x = model_input[:, 0], model_input[:, 1]
        out = model.forward(pt.stack([t, x], dim=1)).squeeze()

        # compute loss
        mse = pt.nn.MSELoss()

        # (dc/dt) - alpha * (d²c / dx²) = 0
        dc_dt = pt.autograd.grad(outputs=out, inputs=t, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()
        dc_dx = pt.autograd.grad(outputs=out, inputs=x, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()
        d2c_dx2 = pt.autograd.grad(outputs=dc_dx, inputs=x, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()

        eq = dc_dt - alpha * d2c_dx2
        loss = mse(eq, pt.zeros(eq.size()))

        return loss

    def compute_loss_prediction(self, model, model_input: pt.Tensor, c_real: pt.Tensor) -> pt.Tensor:
        """
        computes the prediction loss, meaning the loss between true x-value (label) and predicted x

        :param model: the PINN-model
        :param model_input: x- and t-values
        :param c_real: true c-values for given t-, x-values
        :return:  MSE loss between true x-value and prediction
        """
        # make prediction
        out = model.forward(model_input).squeeze()

        mse = pt.nn.MSELoss()
        loss = mse(out, c_real)

        return loss

    def compute_loss_initial_condition(self, model, x: pt.Tensor, t: pt.Tensor) -> pt.Tensor:
        """
        compute the losses for initial- and boundary condition

        :param model: PINN model
        :param x: x-coordinates
        :param t: time values
        :return: losses for fulfilling the initial condition as well as losses for boundary condition
        """
        # make prediction for u(t = 0, x)
        out = model.forward(pt.stack([pt.zeros(t.size()), x])).squeeze()

        # compute loss for u(t = 0, x) = 0
        mse = pt.nn.MSELoss()
        loss = mse(out, pt.zeros(out.size()))

        # since the training routine was implemented for ODE's, it has no boundary condition loss, so just add it here
        # TODO
        loss += model.compute_loss_boundary_condition(model, t, n_x=x.size()[1])
        return loss

    def compute_loss_boundary_condition(self, model, t: pt.Tensor, n_x: int) -> pt.Tensor:
        """
        compute the loss for fulfilling the boundary condition

        :param model: PINN model
        :param t: time values
        :param n_x: number of points in x-direction
        :return: loss for fulfilling the boundary condition(s)
        """
        # make prediction for u(t, x = 0)
        out = model.forward(pt.stack([t, pt.zeros(n_x)])).squeeze()

        # compute loss for u(t, x = 0) = 1
        mse = pt.nn.MSELoss()
        loss = mse(out, pt.ones(out.size()))
        return loss


def compute_analytical_solution(x: pt.Tensor, t: pt.Tensor, alpha: Union[int, float, pt.Tensor] = 1) -> pt.Tensor:
    """
    computes the analytical solution

        .. math:: c(t, x) = 1 - erf(\\frac{x}{\sqrt{4ct}})
        with erf = error function

    to the 1D-diffusion equation
        .. math:: \\frac{\partial c}{\partial t} = \\alpha \\frac{\partial^2 c}{\partial x^2}

    :param x: x-values
    :param t: t-values
    :param alpha: diffusion factor
    :return:  analytical solution c(t, x)
    """
    # TODO: is this correct? -> if u(t = 0, x) = 0 -> frac gives nan due to division by zero
    return 1 - pt.erf(x / pt.sqrt(4 * alpha * t))


def plot_sampled_points():
    pass


def plot_prediction_vs_analytical_solution():
    pass


def wrapper_execute_training():
    pass


if __name__ == "__main__":
    x = pt.linspace(0, 1, 100)
    t = pt.linspace(0.1, 100, 1000)
    mesh_x, mesh_t = pt.meshgrid([x, t], indexing="ij")

    # compute analytical solution
    c = compute_analytical_solution(mesh_x, mesh_t, 1e-3)

    # plot analytical solution
    c_plot = plt.contourf(mesh_x, mesh_t, c, levels=25)
    c_bar = plt.colorbar(c_plot)
    c_bar.set_label("$c$ $\quad[mol / m]$", usetex=True, labelpad=20, fontsize=14)
    plt.xlabel("$x\quad[m]$", fontsize=14, usetex=True)
    plt.ylabel("$t\quad[s]$", fontsize=14, usetex=True)
    plt.show()
