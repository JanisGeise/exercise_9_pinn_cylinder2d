"""
    2. case: solve ODE's using PINN's, here the implementation for:

        .. math:: \\frac{\partial c}{\partial t} = \\alpha \\frac{\partial^2 c}{\partial x^2}

        with:
            alpha = const.

            IC: c(t = 0, x) = 0

            BC: c(t > 0, x = 0) = 1

"""
from utils import *


class PinnDiffusion1D(PINN):
    def compute_loss_equation(self, model, model_input: pt.Tensor, *args) -> pt.Tensor:
        """
        compute the equation loss

        :param model: PINN model
        :param model_input: x- and t-values
        :param args: dict containing t, x and diffusion factor (here only diffusion factor is needed)
        :return: equation loss
        """
        # make prediction: the variable 't' needs to be defined as input, otherwise error since it wouldn't be part of
        # the graph (if the name is not the same...)
        t, x, alpha = model_input[:, 0], model_input[:, 1], args[0]["alpha"]
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
        computes the prediction loss, meaning the loss between true c-value (label) and predicted concentration c

        :param model: the PINN-model
        :param model_input: x- and t-values
        :param c_real: true c-values for given t-, x-values
        :return:  MSE loss between true x-value and prediction
        """
        # make prediction
        out = model.forward(model_input).squeeze()

        mse = pt.nn.MSELoss()
        loss = mse(out, c_real.squeeze())

        return loss

    def compute_loss_initial_condition(self, model, *args) -> pt.Tensor:
        """
        compute the losses for initial- and boundary condition

        :param model: PINN model
        :param args: x-coordinates, and time values for which the equation is evaluated
        :return: losses for fulfilling the initial condition as well as losses for boundary condition
        """
        t, x = args[0]["t"], args[0]["x"]

        # make prediction for u(t = 0, x)
        out = model.forward(pt.stack([pt.zeros(t.size()), x], dim=1)).squeeze()

        # compute loss for u(t = 0, x) = 0
        mse = pt.nn.MSELoss()
        loss = mse(out, pt.zeros(out.size()))

        # since the training routine was implemented for ODE's, it has no boundary condition loss, so just add it here
        loss += model.compute_loss_boundary_condition(model, t, n_x=x.size()[0])
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
        out = model.forward(pt.stack([t, pt.zeros(n_x)], dim=1)).squeeze()

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


def plot_sampled_points(save_path: str, t_pred: pt.Tensor, x_pred: pt.Tensor, t_eq: pt.Tensor, x_eq: pt.Tensor) -> None:
    """
    plot the sampled points for prediction and evaluating the PDE in x- and t-direction

    :param save_path: path to where the plot should be saved
    :param t_pred: sampled t-values for evaluating the model performance (predictions) during training
    :param x_pred: sampled x-values for evaluating the the model performance (predictions) during training
    :param t_eq: sampled t-values for evaluating the PDE during training
    :param x_eq: sampled x-values for evaluating the PDE during training
    :return: None
    """

    plt.scatter(x_pred, t_pred, color="red", marker="o", label="points prediction")
    plt.scatter(x_eq, t_eq, color="green", marker="x", label="points equation")
    plt.xlabel("$x$ $\qquad[m]$", usetex=True, fontsize=14)
    plt.ylabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=0.75, fontsize=10, ncols=2)
    plt.savefig("".join([save_path, f"/plots/sampled_points_1st_pde.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_prediction_vs_analytical_solution(save_path: str, model, mesh_x_ana, mesh_t_ana, c_ana) -> None:
    """
    compare the analytical solution to the solution predicted by the model

    :param save_path: path to the best model as well as where the plot should be saved to
    :param model: the PINN model
    :param mesh_x_ana: grid in x-direction containing all points for evaluating the equation
    :param mesh_t_ana: grid in t-direction containing all points for evaluating the equation
    :param c_ana: analytical solution c(t, x)
    :return: None
    """
    # load the best model and set to eval() mode
    model.load_state_dict(pt.load(save_path + "best_model_train.pt"))
    model.eval()

    # create features for model input and make prediction
    input = pt.stack([mesh_x_ana.flatten(), mesh_t_ana.flatten()], dim=1)
    out = model(input).squeeze().detach().numpy().reshape(mesh_x_ana.size())

    # plot the results
    fig, ax = plt.subplots(1, 2, sharey="row", figsize=(12, 6))
    for p in range(2):
        if p == 0:
            # plot analytical solution
            c_ana = ax[p].contourf(mesh_x_ana, mesh_t_ana, c_ana, levels=25)
            plt.colorbar(c_ana, ax=ax[p])
            ax[p].set_title("$analytical$ $solution$", fontsize=16, usetex=True)
            ax[p].set_ylabel("$t\quad[s]$", fontsize=14, usetex=True)
        else:
            # plot predicted solution
            c_pred = ax[p].contourf(mesh_x_ana, mesh_t_ana, out, levels=25)
            c_pred = plt.colorbar(c_pred, ax=ax[p])
            c_pred.set_label("$c$ $\quad[mol / m]$", usetex=True, labelpad=20, fontsize=14)
            ax[p].set_title("$predicted$ $solution$", fontsize=16, usetex=True)
        ax[p].set_xlabel("$x\quad[m]$", fontsize=14, usetex=True)
    fig.subplots_adjust(hspace=0.05)
    fig.tight_layout()
    plt.savefig("".join([save_path, f"/plots/real_vs_prediction_1st_pde.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def wrapper_execute_training(load_path: str, x_min: Union[int, float] = 0, x_max: Union[int, float] = 1,
                             n_epochs: Union[int, float] = 3000, n_points_pred: int = 10, n_points_eq: int = 75,
                             t_start: Union[int, float] = 0, t_end: Union[int, float] = 1,
                             alpha: Union[int, float] = 1) -> None:
    """
    executes the model training, plots all losses and compares analytical solution to the predicted one

    :param load_path: path where everything should be saved to / loaded from
    :param x_min: strat x-value
    :param x_max: end x-value
    :param n_epochs: number of epochs to run the training
    :param n_points_pred: number of points for which the feature-label pairs should be evaluated during training
    :param n_points_eq: number of points for which the PDE should be evaluated during training
    :param t_start: start time
    :param t_end: end time
    :param alpha: diffusion coefficient
    :return: None
    """
    # instantiate model: we want to predict an c(t, x) for a given t-value and x-value
    pinn = PinnDiffusion1D(n_inputs=2, n_outputs=1, n_layers=4, n_neurons=25)

    # use lhs to sample points for prediction and equation in given bounds
    t_eq, x_eq = lhs_sampling([t_start, x_min], [t_end, x_max], n_samples=n_points_eq)
    t_pred, x_pred = lhs_sampling([t_start, x_min], [t_end, x_max], n_samples=n_points_pred)

    # plot the sampled points
    plot_sampled_points(load_path, t_pred, x_pred, t_eq, x_eq)

    # generate feature-label pairs
    feature_pred, label_pred = pt.stack([t_pred, x_pred], dim=1), compute_analytical_solution(x_pred, t_pred, alpha)
    feature_eq, label_eq = pt.stack([t_eq, x_eq], dim=1), compute_analytical_solution(x_eq, t_eq, alpha)

    # train model using the sampled points
    losses = train_pinn(pinn, feature_pred.requires_grad_(True), label_pred.unsqueeze(-1).requires_grad_(True),
                        feature_eq.requires_grad_(True), label_eq.unsqueeze(-1).requires_grad_(True), epochs=n_epochs,
                        save_path=load_path, equation_params={"alpha": alpha, "t": t_eq, "x": x_eq})

    # plot the losses
    plot_losses(load_path, losses, case="1st_pde")

    # compare analytical solution against the predicted one
    mesh_x, mesh_t = pt.meshgrid([pt.linspace(x_min, x_max, 50), pt.linspace(t_start, t_end, 100)], indexing="ij")

    # compute analytical solution
    c = compute_analytical_solution(mesh_x, mesh_t, 1e-3)
    plot_prediction_vs_analytical_solution(load_path, pinn, mesh_x, mesh_t, c)


if __name__ == "__main__":
    pass
