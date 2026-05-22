# Quantitative Research & Modeling

## Overview
The `research` module serves as the quantitative laboratory of the project. It contains standalone scripts for mathematical modeling, theoretical pricing, and historical calibration of complex financial derivatives.

## Core Models & Scripts

### Stochastic Volatility & Jump-Diffusion
- **`Heston.py`**: Implements the Heston stochastic volatility model. It uses Fourier inversion techniques and numerical integration (Simpson's rule) to compute analytical European option prices. Includes functionality to calibrate the model to market data to extract implied parameters ($v_0, \kappa, \theta, \rho, \sigma$).
- **`Bates.py` & `Bates_IV_Calibration.py`**: Extends the Heston model by incorporating Merton's log-normal jump-diffusion (the Bates model). These scripts handle the robust calibration of both continuous volatility parameters and jump components ($\lambda, \mu_J, \sigma_J$) to fit the market implied volatility surface accurately.

### Simulation & Numerical Methods
- **`heston_models_comparaison.py`**: Provides a Monte Carlo simulation engine for pricing options under the Heston dynamics using Euler-Maruyama discretization. It also implements a Finite Differences solver using the Modified Craig-Sneyd (MCS) scheme, offering a robust numerical alternative for solving the Heston PDE.

## Key Quantitative Concepts
- **Characteristic Functions & Fourier Inversion**: Used for fast, analytical pricing of options without relying solely on computationally expensive simulations.
- **Differential Evolution**: A stochastic population-based optimization algorithm used to minimize the Vega-weighted Root Mean Square Error (RMSE) during model calibration, avoiding local minima.
- **Implied Volatility Surfaces**: 3D visualization of the market's expectation of future volatility across different strikes and maturities.

