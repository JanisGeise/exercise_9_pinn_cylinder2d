"""
    2. case: solve ODE's using PINN's, here the implementation for:

        .. math:: \\frac{dx}{dt} = -kx

            x(t=0) = 1

            k \\ne const.

"""
from utils import *
from exponential_decay_const_k import compute_analytical_solution


class PinnVariableK(PINN):
    def compute_loss_equation(self, model, model_input: pt.Tensor, *args) -> pt.Tensor:
        """
        computes the MSE loss of the ODE using the predicted x-value for a given t- and k-value, in contrast to
        k = const., here k is input to model

        :param model: the PINN-model
        :param model_input: points in time + 1 point for decay factor k each
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

    def compute_loss_initial_condition(self, model, n_k: int) -> pt.Tensor:
        """
        compute the loss for the initial condition

        :param model: the PINN-model
        :param n_k: number of k-values in batch
        :return: MSE loss for initial condition
        """
        # make prediction
        out = model.forward(pt.zeros((n_k, model.n_inputs))).squeeze()
        mse = pt.nn.MSELoss()

        # x(t = 0) = 1 for all k
        loss = mse(out, pt.ones(out.size()).squeeze())

        return loss

    def compute_loss_boundary_condition(self, *args):
        pass


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
        plt.plot(t_eq, x_eq[:, k], color="green", linestyle="none", marker="x")
        plt.plot(t_pred[k, :], x_pred[k, :], color="red", linestyle="none", marker="o")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/sampled_points_2nd_ode.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_prediction_vs_analytical_solution(save_path: str, load_path, model, t, x, k, k_test) -> None:
    """
    plot the predicted solution x(t) against the analytical one

    :param save_path: path to the directory where the plot should be saved in
    :param load_path: path to the model
    :param model: the pinn-model
    :param t: time values
    :param x: x(t)-values of the analytical solution
    :param k: k-values
    :param k_test: indices of the k-values the PINN was trained on
    :return: None
    """
    # load the best model and set to eval() mode
    model.load_state_dict(pt.load(load_path + "best_model_train.pt"))
    model.eval()

    # predict the solution x(t) based on given t
    x_pred = pt.zeros(x.size())
    for idx, time in enumerate(t):
        feature = pt.stack([time, k], dim=1)
        x_pred[idx, :] = model(feature).detach().squeeze()

    # plot analytical solution vs. predicted one
    for idx in range(x.size()[1]):
        if idx == 0:
            if idx not in k_test:
                plt.plot(t[:, idx], x[:, idx], color="black", linestyle="--", label="analytical solution")
                plt.plot(t[:, idx], x_pred[:, idx], color="red", linestyle="--", label="predicted solution")
            else:
                plt.plot(t[:, idx], x[:, idx], color="black", label="analytical solution")
                plt.plot(t[:, idx], x_pred[:, idx], color="red", label="predicted solution")
        else:
            if idx not in k_test:
                plt.plot(t[:, idx], x[:, idx], color="black", linestyle="--")
                plt.plot(t, x_pred[:, idx], color="red", linestyle="--")
            else:
                plt.plot(t[:, idx], x[:, idx], color="black")
                plt.plot(t, x_pred[:, idx], color="red")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/prediction_vs_analytical_solution_variable_k.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def wrapper_execute_training(load_path: str, k: Union[list, pt.Tensor], n_epochs: Union[int, float] = 1000,
                             t_end: Union[int, float] = 10, n_points_eq: int = 25, n_points_pred: int = 5) -> None:
    """
    manages the execution of generating and sampling data for the model-training as well as plotting losses, making
    predictions and plotting the predictions against the analytical solution

    :param load_path: path where the plots & models should be saved in
    :param k: list or tensor with decay factors for the ODE
    :param n_epochs: number of epochs to run the training
    :param t_end: last time step for computing / predicting the solution
    :param n_points_eq: number of points for which the equation should be evaluated during training (loss equation)
    :param n_points_pred: number of points for which the prediction should be evaluated during training (loss pred.)
    :return: None
    """
    # instantiate model: we want to predict an x(t) for a given t and k
    pinn = PinnVariableK(n_inputs=2, n_outputs=1, n_layers=3, n_neurons=25)

    # compute analytical solution as comparison
    x, t = pt.zeros((1000, len(k))), pt.zeros((1000, len(k)))
    for idx, val in enumerate(k):
        x[:, idx], t[:, idx] = compute_analytical_solution(t_end=t_end, k=val)

    # use min- / max and one value in between for training / testing and the other k-values to test model on unseen data
    idx_train = pt.multinomial(k, num_samples=len(k)-3)       # poor accuracy when extrapolating
    # idx_train = pt.multinomial(k, num_samples=len(k))
    k_train = k[idx_train]

    # use latin hypercube sampling to sample points as feature-label pairs for prediction
    label_pred, feature_pred = pt.zeros((len(k), n_points_pred)), pt.zeros((len(k), n_points_pred, 2))
    for idx, val in enumerate(k_train):
        # sample time steps in interval [t_start, t_end] for each k-value
        t_pred = lhs_sampling([0], [t_end], n_points_pred).squeeze()

        # feature = [sampled time steps, k-value] for each k-value
        feature_pred[idx, :] = pt.stack((t_pred, pt.ones(t_pred.size()) * val), dim=1)

        # label = [N_k_values, corresponding x-values] for each k-value
        label_pred[idx, :] = pt.exp(-val * t_pred)

    # sample points for which the equation should be evaluated
    feature_eq = lhs_sampling([0, pt.min(k_train)], [t_end, pt.max(k_train)], n_points_eq).transpose(0, 1)
    label_eq = pt.stack([pt.exp(-k_val * feature_eq[:, 0]) for k_val in k_train], dim=1)

    # plot sampled points
    plot_sampled_points_var_k(load_path, t, x[:, idx_train], feature_pred[:, :, 0], label_pred,
                              feature_eq[:, 0].unsqueeze(-1), label_eq)

    # train model using the sampled points
    losses = train_pinn(pinn, feature_pred.requires_grad_(True), label_pred.requires_grad_(True),
                        feature_eq.requires_grad_(True), label_eq.requires_grad_(True), epochs=n_epochs,
                        save_path=load_path, equation_params=int(k.size()[0]))

    # plot the losses
    plot_losses(load_path, losses, case="2nd_ode")

    # plot the analytical solutions against the predicted ones
    plot_prediction_vs_analytical_solution(load_path, load_path, pinn, t, x, k, idx_train)


if __name__ == "__main__":
    pass
