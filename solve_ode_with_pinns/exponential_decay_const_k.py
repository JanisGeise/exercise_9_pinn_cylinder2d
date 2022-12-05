"""
    1. case: solve ODE's using PINN's, here the implementation for:

        dx/dt = -kx     with    x(t=0) = 1, k = const.

"""
import matplotlib.pyplot as plt

from utils import *


class PinnConstK(PINN):
    def compute_loss_equation(self, model, t: pt.Tensor, k: Union[int, float]) -> pt.Tensor:
        """
        computes the MSE loss of the ODE using the predicted x-value for a given t- and k-value

        :param model: the PINN-model
        :param t: points in time
        :param t: time-value used for predicting the x-value
        :param k: decay factor
        :return: MSE-loss
        """
        t = t.squeeze(0)
        # make prediction
        out = model.forward(t).squeeze()

        # compute loss
        mse = pt.nn.MSELoss()

        # ODE: dx/dt + kx = 0
        dx_dt = pt.autograd.grad(outputs=out, inputs=t, create_graph=True, grad_outputs=pt.ones_like(out))[0].squeeze()
        loss = mse(dx_dt + k * out, pt.zeros(dx_dt.size()))

        return loss

    def compute_loss_prediction(self, model, t: pt.Tensor, x_real: pt.Tensor) -> pt.Tensor:
        """
        computes the prediction loss, meaning the loss between true x-value (label) and predicted x

        :param model: the PINN-model
        :param t: points in time
        :param x_real: true x-value
        :return:  MSE loss between true x-value and prediction
        """
        # make prediction
        out = model.forward(t.squeeze(0)).squeeze()

        mse = pt.nn.MSELoss()
        loss = mse(out, x_real.squeeze())

        return loss


def compute_analytical_solution(t_start: int = 0, t_end: int = 1, k: Union[int, float] = 1, c: int = 0,
                                n_points: int = 1000) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    computes the analytical solution to the ODE
        dx/dt = -kx;    x(t=0) = 1, k = const.

    :param t_start: start time
    :param t_end: end time
    :param k: value for parameter "k" (factor for decay rate)
    :param n_points: number of points for which the solution should be calculated
    :param c: initial condition for x(t=0)
    :return: x(t) and corresponding t as tensor
    """
    # analytical solution: x(t) = exp(-kt)
    t = pt.linspace(t_start, t_end, n_points)
    x = pt.exp(-k * t) + c

    return x, t


def plot_sampled_points(save_path: str, t_real, x_real, t_pred, x_pred, t_eq, x_eq) -> None:
    """


    :param save_path:
    :param t_real:
    :param x_real:
    :param t_pred:
    :param x_pred:
    :param t_eq:
    :param x_eq:
    :return:
    """
    plt.plot(t_real, x_real, color="black", label="analytical solution")
    plt.plot(t_pred, x_pred, color="red", linestyle="none", marker="o")
    plt.plot(t_eq, x_eq, color="green", linestyle="none", marker="x")

    # dummy points for legend
    plt.plot(t_pred[0], x_pred[0], color="red", linestyle="none", marker="o", label="points prediction")
    plt.plot(t_eq[0], x_eq[0], color="green", linestyle="none", marker="x", label="points equation")

    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/sampled_points.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_prediction_vs_analytical_solution(save_path: str, load_path, model, t, x) -> None:
    """

    :param save_path:
    :param load_path:
    :param model:
    :param t:
    :param x:
    :return:
    """
    # load the best model and set to eval() mode
    model.load_state_dict(pt.load(load_path + "best_model_train.pt"))
    model.eval()

    # predict the solution x(t) based on given t
    x_pred = pt.zeros(x.size())

    # initial condition
    x_pred[0] = 1
    for time in range(1, len(t)):
        x_pred[time] = model(t[time].unsqueeze(-1)).detach().squeeze()

    # plot analytical solution vs. predicted one
    plt.plot(t, x, color="black", label="analytical solution")
    plt.plot(t, x_pred, color="red", label="predicted solution")
    plt.xlabel("$t$ $\qquad[s]$", usetex=True, fontsize=14)
    plt.ylabel("$x(t)$ $\qquad[-]$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10)
    plt.savefig("".join([save_path, f"/plots/prediction_vs_analytical_solution_const_k.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def plot_losses(savepath: str, loss: Tuple[list, list, list]) -> None:
    """
    plot training- and equation- and prediction losses

    :param savepath: path where the plot should be saved
    :param loss: tensor containing all the losses
    :return: None
    """
    # plot training- and validation losses
    plt.plot(range(len(loss[0])), loss[0], color="blue", label="total training loss")
    plt.plot(range(len(loss[1])), loss[1], color="green", label="equation loss")
    plt.plot(range(len(loss[2])), loss[2], color="red", label="prediction loss")
    plt.xlabel("$epoch$ $number$", usetex=True, fontsize=14)
    plt.ylabel("$MSE$", usetex=True, fontsize=14)
    plt.legend(loc="upper right", framealpha=1.0, fontsize=10, ncols=1)
    plt.yscale("log")
    plt.savefig("".join([savepath, f"/plots/losses.png"]), dpi=600)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


def wrapper_execute_training(load_path: str, k: Union[int, float], n_epochs: Union[int, float]) -> None:
    """


    :param load_path:
    :param k:
    :param n_epochs:
    :return:
    """
    # instantiate model: we want to predict an x(t) for a given t
    pinn = PinnConstK(n_inputs=1, n_outputs=1, n_layers=5, n_neurons=75)

    # compute analytical solution as comparison
    x, t = compute_analytical_solution(t_end=10, k=k)

    # sample some "real" points from the analytical solution as training- and validation data
    label_pred, feature_pred = compute_analytical_solution(t_end=10, n_points=5, k=k)

    # use latin hypercube sampling to sample points as feature-label pairs for prediction (still not working...)
    # feature_pred = lhs_sampling([0], [10], 5)
    # label_pred = pt.exp(-k * feature_pred)

    # sample points for which the equation should be evaluated
    feature_eq = lhs_sampling([0], [10], 50)
    label_eq = pt.exp(-k * feature_eq)

    # plot sampled points
    plot_sampled_points(load_path, t, x, feature_pred, label_pred, feature_eq, label_eq)

    # train model using the sampled points
    losses = train_pinn(pinn, feature_pred.unsqueeze(-1).requires_grad_(True),
                        label_pred.unsqueeze(-1).requires_grad_(True), feature_eq.unsqueeze(-1).requires_grad_(True),
                        label_eq.unsqueeze(-1).requires_grad_(True), equation_params=k, epochs=n_epochs,
                        save_path=load_path)

    # plot the losses
    plot_losses(load_path, losses)

    # predict solution and compare to analytical solution
    plot_prediction_vs_analytical_solution(save_path=load_path, load_path=load_path, model=pinn, t=t, x=x)


if __name__ == "__main__":
    pass
