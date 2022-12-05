# Exercise 9 - Approximating the flow past a cylinder with limited data

This code is a modified version of the code presented in
[PINN-for-NS-equation](https://github.com/Shengfeng233/PINN-for-NS-equation) by
[Shengfeng233](https://github.com/Shengfeng233). The modyfied code is related to the
[ML in CFD lecture](https://github.com/AndreWeiner/ml-cfd-lecture) by [Andre Weiner](https://github.com/AndreWeiner).

## Flow past a cylinder
The implementation for solving the Navier-Stokes-equations for a flow passt a cylinder is not working properly at the
moment. The directory *code* contains all scripts related to the *cylinder2D*-case while the
*cylinder2D_quasi_steady_flow_field.pkl* provides the flow field as training data and comparison to the predicted flow
field.

**Note**: The *main.py* script can be executed without any issues (just the results / predictions are not accurate at
        the moment). However, the script *train_models_minimal_case.py* will not yield any results due to *nan* values
        when calculating the gradients and losses.


## Simpel ODE's
As an alternative, the directory *solve_ode_with_pinns* contains scripts for solving the following ODE's using PINN's:

1. ${dx \over dt} = -kx$; $k = const.$, $x(t = 0) = 1$  
2. ${dx \over dt} = -kx$; $k \ne const.$, $x(t = 0) = 1$
