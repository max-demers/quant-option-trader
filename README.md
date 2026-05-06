# Quantitative Options Trading Framework

## Overview
This repository contains a comprehensive algorithmic trading framework specialized in options pricing, volatility modeling, and automated trading. The project bridges the gap between theoretical quantitative finance and practical algorithmic execution, utilizing advanced stochastic volatility and jump-diffusion models to identify trading opportunities in the options market.

## Project Architecture
The codebase is modularized into two distinct environments to separate experimentation from execution:

- **`research/`**: Dedicated to quantitative research, mathematical modeling, and algorithm development. This includes the theoretical pricing engines, Monte Carlo simulations, and historical calibration scripts.
- **`trading_bot/`**: The production environment containing the automated execution system. It handles live market data ingestion, continuous model recalibration, portfolio management, and delta hedging.

## Key Features
- **Advanced Pricing Models**: Implementation of the **Heston** (stochastic volatility) and **Bates** (stochastic volatility with jump-diffusion) models.
- **Robust Calibration**: Uses global optimization techniques (`scipy.optimize.differential_evolution`) to calibrate model parameters against real-world market implied volatility surfaces.
- **Automated Execution**: A scheduled trading bot that continuously scans for statistical arbitrage opportunities (mispriced options) based on theoretical model values.
- **Risk Management**: Dynamic portfolio tracking with automated Delta Hedging to maintain market neutrality.

## Installation & Setup
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the research scripts to visualize calibrations, or start the trading bot for live market scanning:
   ```bash
   python -m Project.trading_bot.trader
   ```

## Disclaimer
> [!WARNING]
> **Not Financial Advice.** This software is provided for educational and research purposes only. Trading options and other financial derivatives involves significant risk of loss and is not suitable for all investors. If you choose to use this framework or any of its algorithms with real money, you do so entirely at your own risk. The authors and contributors are not responsible for any financial losses or damages incurred.
