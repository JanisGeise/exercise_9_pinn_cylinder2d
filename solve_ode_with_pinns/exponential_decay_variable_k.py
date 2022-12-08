"""
    2. case: solve ODE's using PINN's, here the implementation for:

        dx/dt = -kx     with    x(t=0) = 1, k != const.

"""
from matplotlib import pyplot as plt

from utils import *
from exponential_decay_const_k import compute_analytical_solution, plot_losses


class PinnVariableK(PINN):
    def compute_loss_equation(self, model, model_input: pt.Tensor) -> pt.Tensor:
        """
        computes the MSE loss of the ODE using the predicted x-value for a given t- and k-value, in contrast to
        k = const., here k is input to model

        :param model: the PINN-model
        :param model_input: points in time + 1 point for decay factor k each
        :param t: time-value used for predicting the x-value
        :return: MSE-loss
        """
        # make prediction: the variable 't' needs to be defined as input, otherwise error since it wouldn't be part of
        # the graph (if the name is not the same...)
        t, k = model_input[:, 0], model_input[:, 1]
        out = model.forward(pt.stack([t, k], dim=1)).squeeze()

        # compute loss
        mse = pt.nn.MSELoss()

        # ODE: dx/dt + kx = 0
        dx_dt = pt.autograd.grad(outputs=out, inputs=t, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()
        loss = mse(dx_dt + k * out, pt.zeros(dx_dt.size()))

        return loss

    def compute_loss_prediction(self, model, model_input: pt.Tensor, x_real: pt.Tensor) -> pt.Tensor:
        """
        computes the prediction loss, meaning the loss between true x-value (label) and predicted x

        :param model: the PINN-model
        :param model_input: points in time + 1 point for decay factor k each
        :param x_real: true x-value
        :return:  MSE loss between true x-value and prediction
        """
        # make prediction
        out = model.forward(model_input).squeeze()

        mse = pt.nn.MSELoss()
        loss = mse(out, x_real)

        return loss


def plot_sampled_points_var_k(save_path: str, t_real, x_real, t_pred, x_pred, t_eq, x_eq) -> None:
    """
    plot the points sampled for making predictions and the points for evaluating the ODE as well as the corresponding
    analytical solution

    :param save_path: path to the directory where the plot should be saved in
    :param t_real: time values from the analytical solution
    :param x_real: corresponding x(t)-values from the analytical solution
    :param t_pred: time values sampled for the predictions
    :param x_pred: corresponding x(t)-values for the predictions
    :param t_eq: time values sampled for evaluating the equation
    :param x_eq: corresponding x(t)-values for the equation
    :return: None
    """
    # dummy points for legend
    plt.plot(t_real[:, 0], x_real[:, 0], color="black", label="analytical solution")
    plt.plot(t_pred[0, 0], x_pred[0, 0], color="red", linestyle="none", marker="o", label="points prediction")
    plt.plot(t_eq[0, 0], x_eq[0, 0], color="green", linestyle="none", marker="x", label="points equation")

    for k in range(x_real.size()[1]):
        plt.plot(t_real[:, k], x_real[:, k], color="black")
        plt.plot(t_pred, x_pred[:, k], color="red", linestyle="none", marker="o")
        plt.plot(t_eq, x_eq[:, k], color="green", linestyle="none", marker="x")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/sampled_points_2nd_ode.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_prediction_vs_analytical_solution(save_path: str, load_path, model, t, x, k) -> None:
    """
    plot the predicted solution x(t) against the analytical one

    :param save_path: path to the directory where the plot should be saved in
    :param load_path: path to the model
    :param model: the pinn-model
    :param t: time values
    :param x: x(t)-values of the analytical solution
    :param x: k-values
    :return: None
    """
    # load the best model and set to eval() mode
    model.load_state_dict(pt.load(load_path + "best_model_train.pt"))
    model.eval()

    # predict the solution x(t) based on given t
    x_pred = pt.zeros(x.size())
    for idx, time in enumerate(t):
        if idx == 0:
            # initial condition
            x_pred[idx, :] = pt.ones(x_pred.size()[-1])
        else:
            feature = pt.stack([time, k], dim=1)
            x_pred[idx, :] = model(feature).detach().squeeze()

    # plot analytical solution vs. predicted one
    for k in range(x.size()[1]):
        if k == 0:
            plt.plot(t[:, k], x[:, k], color="black", label="analytical solution")
            plt.plot(t, x_pred[:, k], color="red", label="predicted solution")
        else:
            plt.plot(t[:, k], x[:, k], color="black")
            plt.plot(t, x_pred[:, k], color="red")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/prediction_vs_analytical_solution_variable_k.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def wrapper_execute_training(load_path: str, k: Union[list, pt.Tensor], n_epochs: Union[int, float]) -> None:
    """
    manages the execution of generating and sampling data for the model-training as well as plotting losses, making
    predictions and plotting the predictions against the analytical solution

    :param load_path: path where the plots & models should be saved in
    :param k: list or tensor with decay factors for the ODE
    :param n_epochs: number of epochs to run the training
    :return: None
    """
    # instantiate model: we want to predict an x(t) for a given t
    pinn = PinnVariableK(n_inputs=2, n_outputs=1, n_layers=5, n_neurons=75)

    # compute analytical solution as comparison
    x, t = pt.zeros((1000, len(k))), pt.zeros((1000, len(k)))
    for idx, val in enumerate(k):
        x[:, idx], t[:, idx] = compute_analytical_solution(t_end=10, k=val)

    # sample some "real" points from the analytical solution as training- and validation data (for now)
    label_pred = pt.zeros((5, len(k)))
    for idx, val in enumerate(k):
        label_pred[:, idx], t_pred = compute_analytical_solution(t_end=10, n_points=5, k=val)

    # feature has [time value, k-value], for all time values
    feature_pred = pt.stack([t_pred, k], dim=1)

    # use latin hypercube sampling to sample points as feature-label pairs for prediction (still not working...)
    # feature_pred = lhs_sampling([0], [10], 5)
    # label_pred = pt.exp(-k * feature_pred)

    # sample points for which the equation should be evaluated
    feature_eq = lhs_sampling([0, k[0]], [10, k[-1]], 100).transpose(0, 1)
    label_eq = pt.stack([pt.exp(-k_val * feature_eq[:, 0]) for k_val in k], dim=1)

    # plot sampled points
    plot_sampled_points_var_k(load_path, t, x, feature_pred[:, 0].unsqueeze(-1), label_pred,
                              feature_eq[:, 0].unsqueeze(-1), label_eq)

    # train model using the sampled points
    feature_pred = feature_pred.unsqueeze(0).expand((label_pred.size()[0], label_pred.size()[1], feature_pred.size()[1]))

    losses = train_pinn(pinn, feature_pred.requires_grad_(True), label_pred.requires_grad_(True),
                        feature_eq.requires_grad_(True), label_eq.requires_grad_(True), epochs=n_epochs,
                        save_path=load_path)

    # plot the losses
    plot_losses(load_path, losses, case="2nd_ode")

    # plot the analytical solutions against the predicted ones
    plot_prediction_vs_analytical_solution(load_path, load_path, pinn, t, x, k)


if __name__ == "__main__":
    pass
