# Exercise 9 - Approximating the flow past a cylinder with limited data

This code for the flow past a cylinder is located in the *code* directory. This code is a modified version of the code presented in
[PINN-for-NS-equation](https://github.com/Shengfeng233/PINN-for-NS-equation) by
[Shengfeng233](https://github.com/Shengfeng233). The modyfied code is related to the
[ML in CFD lecture](https://github.com/AndreWeiner/ml-cfd-lecture) by [Andre Weiner](https://github.com/AndreWeiner).

## Flow past a cylinder
The implementation for solving the Navier-Stokes-equations for a flow past a cylinder is not working properly at the
moment. The directory *code* contains all scripts related to the *cylinder2D*-case while the
*cylinder2D_quasi_steady_flow_field.pkl* provides the flow field as training data and comparison to the predicted flow
field. The directory *solve_ode_with_pinns* contains scripts for solving simple ODE's & PDE's with PINN's as described below.

**Note**: The *main.py* script can be executed without any issues (just the results / predictions are not accurate at
        the moment). The script *train_models_minimal_case.py* works in general as well, but at the moment the model is
        not learning anything (prediction loss remains constant). Further, the runtimes for training the model are very high.


## Simpel ODE's / PDE's
As an alternative, the directory *solve_ode_with_pinns* contains scripts for solving the following ODE's using PINN's:

1. exponential decay:  
   1.1. ${dx \over dt} = -kx$; $k = const.$, $x(t = 0) = 1$  
   1.2. ${dx \over dt} = -kx$; $k \ne const.$, $x(t = 0) = 1$
2. diffusion equation (1D):  
        ${\partial c \over  \partial t} = \alpha {\partial^2 c \over \partial x^2}$; $\alpha = const.$, $x(t = 0, x) = 0$,
        $x(t > 0, x = 0) = 1$
