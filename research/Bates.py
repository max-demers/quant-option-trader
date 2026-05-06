import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

# Import the new modular classes
from Project.trading_bot.market_data_handler import MarketDataHandler
from Project.trading_bot.bates_model import BatesModel


# Options d'affichage pour Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def plot_calibration_results(market_data, bates_model, S0, r):
    """
    Generates and optionally displays a plot comparing market prices to calibrated model prices.
    """
    print("\nGenerating comparison plot...")
    all_model_prices_opt = []
    for T, group in market_data.groupby('T'):
        strikes = group['strike'].values
        # Use the bates_model instance to price
        model_prices_T, _ = bates_model.bates_pricer_robust(S0, strikes, T, r)
        all_model_prices_opt.append(pd.Series(model_prices_T, index=group.index))

    market_data['model_price'] = pd.concat(all_model_prices_opt)

    unique_expiries = sorted(market_data['T'].unique())
    num_expiries = len(unique_expiries)

    fig, axes = plt.subplots(1, num_expiries, figsize=(6 * num_expiries, 5), squeeze=False)
    fig.suptitle('Market Prices vs. Calibrated Bates Model', fontsize=16)

    for i, T_val in enumerate(unique_expiries):
        ax = axes[0, i]
        expiry_data = market_data[market_data['T'] == T_val].sort_values(by='strike')

        ax.plot(expiry_data['strike'], expiry_data['price'], 'o-', label='Market Prices')
        ax.plot(expiry_data['strike'], expiry_data['model_price'], 'x--', label='Bates Model')

        ax.set_title(f'Expiry T = {T_val:.3f} years (~{int(T_val * 365.25)} days)')
        ax.set_xlabel('Strike Price (K)')
        ax.set_ylabel('Option Price')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if input("Display calibration comparison plot? (y/n): ").lower() == 'y':
        plt.show()

def plot_iv_surface(iv_data, bates_model, S0, r):
    """
    Generates and displays a 3D plot of the Implied Volatility surface.
    """
    print("Generating 3D Implied Volatility Surface plot (opening in browser)...")
    print("This involves pricing a dense grid and can be intensive.")
    
    # 1. Define a dense grid for the smooth surface
    min_T, max_T = iv_data['T'].min(), iv_data['T'].max()
    min_K, max_K = iv_data['strike'].min(), iv_data['strike'].max()
    T_dense = np.linspace(min_T, max_T, 50)
    K_dense = np.arange(int(min_K), int(max_K), 1.0)

    grid_data = []
    # 2. Loop through grid and calculate model's implied volatility
    for t_val in tqdm(T_dense, desc="Pricing Grid for 3D Surface"):
        model_prices_t, _ = bates_model.bates_pricer_robust(S0, K_dense, t_val, r)
        for i, k_val in enumerate(K_dense):
            iv = bates_model.implied_volatility(model_prices_t[i], S0, k_val, t_val, r)
            if not np.isnan(iv):
                grid_data.append({'T': t_val, 'strike': k_val, 'model_iv': iv})
    
    model_iv_df = pd.DataFrame(grid_data)

    if not model_iv_df.empty:
        # 3. Reshape data for plotting
        IV_surface_model = model_iv_df.pivot_table(index='strike', columns='T', values='model_iv').values
        K_pivoted = model_iv_df.pivot_table(index='strike', columns='T', values='model_iv').index.values
        T_pivoted = model_iv_df.pivot_table(index='strike', columns='T', values='model_iv').columns.values

        # 4. Create Plotly traces
        surface_trace = go.Surface(
            x=K_pivoted, y=T_pivoted, z=IV_surface_model.T,
            colorscale='Viridis', opacity=0.9, name='Calibrated Bates IV Surface'
        )
        scatter_trace = go.Scatter3d(
            x=iv_data['strike'], y=iv_data['T'], z=iv_data['implied_vol'],
            mode='markers', marker=dict(size=3, color='red', symbol='circle'), name='Market IV'
        )
        layout = go.Layout(
            title='Implied Volatility Surface: Calibrated Model vs. Market',
            scene=dict(
                xaxis_title='Strike Price (K)', yaxis_title='Time to Maturity (T)', zaxis_title='Implied Volatility (BS)',
                zaxis_range=[iv_data['implied_vol'].min(), iv_data['implied_vol'].max()]
            )
        )
        fig = go.Figure(data=[surface_trace, scatter_trace], layout=layout)
        fig.show()
    else:
        print("Could not generate the model's IV surface.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    
    # --- Configuration ---
    TICKER_SYMBOL = "AAPL"
    CALIBRATION_TARGETS_DAYS = [7, 14, 45, 365]
    CALIBRATION_STRIKE_WINDOW_PCT = 0.08
    CALIBRATION_MIN_OPTIONS_PER_EXPIRY = 8
    CALIBRATION_MIN_VOLUME = 10
    
    SCAN_CONFIG = {
        'strike_window_pct': 0.08,
        'min_volume': 50,
        'min_open_interest': 100,
        'min_days_to_expiry': 18,
        'max_days_to_expiry': 180,
    }

    # Optimizer settings
    OPTIMIZER_BOUNDS = [
        (0.001, 0.9), (1.0, 8.0), (0.01, 2.0), (-0.9, -0.3), (0.1, 1.5), # Heston
        (0.01, 2.0), (-0.5, 0.1), (0.0001, 0.5)                        # Jumps
    ]
    OPTIMIZER_SETTINGS = {
        'maxiter': 2000, 'popsize': 15, 'tol': 0.01, 'updating': 'deferred', 'workers': -1
    }

    # --- 1. Initialize Handlers ---
    market_handler = MarketDataHandler(TICKER_SYMBOL)
    bates_model = BatesModel()

    try:
        # --- 2. Fetch Data for Calibration ---
        S0_market = market_handler.get_current_price()
        r_market = market_handler.get_risk_free_rate()
        
        calibration_data = market_handler.fetch_calibration_data(
            S0=S0_market,
            targets_in_days=CALIBRATION_TARGETS_DAYS,
            strike_window_pct=CALIBRATION_STRIKE_WINDOW_PCT,
            min_options=CALIBRATION_MIN_OPTIONS_PER_EXPIRY,
            min_volume=CALIBRATION_MIN_VOLUME
        )

        if calibration_data is None or calibration_data.empty:
            raise ValueError("Failed to fetch sufficient data for calibration.")

        # --- 3. Calibrate the Model ---
        print(f"\nCalibrating to {len(calibration_data)} options across {calibration_data['T'].nunique()} maturities.")
        print(f"Underlying Price (S0): {S0_market:.2f}")
        print(f"Risk-free rate (r): {r_market:.3f}")
        
        bates_model.calibrate(calibration_data, S0_market, r_market, OPTIMIZER_BOUNDS, OPTIMIZER_SETTINGS)

        if bates_model.rmse is None:
             raise RuntimeError("Calibration failed, cannot proceed.")
        
        # --- 4. Visualize Calibration and Scan for Opportunities ---
        plot_calibration_results(calibration_data, bates_model, S0_market, r_market)

        # --- Full Market Scan for Trading Opportunities ---
        tradable_options = market_handler.fetch_all_tradable_options(S0_market, SCAN_CONFIG)
        
        if tradable_options is not None and not tradable_options.empty:
            print("Pricing all found options with the calibrated Bates parameters...")
            all_model_prices_scan = []
            all_deltas_scan = []
            for T_scan, group in tradable_options.groupby('T'):
                strikes_scan = group['strike'].values
                model_prices_T_scan, deltas_T_scan = bates_model.bates_pricer_robust(S0_market, strikes_scan, T_scan, r_market)
                all_model_prices_scan.append(pd.Series(model_prices_T_scan, index=group.index))
                all_deltas_scan.append(pd.Series(deltas_T_scan, index=group.index))

            tradable_options['model_price'] = pd.concat(all_model_prices_scan)
            tradable_options['delta'] = pd.concat(all_deltas_scan)
            tradable_options['price_diff'] = tradable_options['model_price'] - tradable_options['price']
            tradable_options['abs_price_diff'] = np.abs(tradable_options['price_diff'])
            tradable_options['spread_cost'] = (tradable_options['ask'] - tradable_options['bid']) / 2
            tradable_options['theo_profit'] = tradable_options['abs_price_diff'] - tradable_options['spread_cost']

            # Filter for significant discrepancies
            outlier_threshold = 3 * bates_model.rmse
            significant_discrepancies = tradable_options[tradable_options['theo_profit'] > outlier_threshold]
            
            print(f"\n--- Options with Significant Theoretical Profit (> 3 * RMSE = ${outlier_threshold:.2f}) ---")
            if not significant_discrepancies.empty:
                display_cols = ['T', 'strike', 'price', 'model_price', 'price_diff', 'theo_profit', 'delta', 'volume', 'openInterest']
                print(significant_discrepancies.sort_values(by='theo_profit', ascending=False)[display_cols].round(4))
            else:
                print("No options found with theoretical profit greater than three times the calibration RMSE.")

            # --- Implied Volatility Surface ---
            if input("\nCalculate and display Implied Volatility surface? (y/n): ").lower() == 'y':
                iv_data = tradable_options.copy()
                print("Calculating Implied Volatility for all scanned options...")
                iv_data['implied_vol'] = iv_data.apply(
                    lambda row: bates_model.implied_volatility(row['price'], S0_market, row['strike'], row['T'], r_market),
                    axis=1
                ).dropna()

                if not iv_data.empty:
                    plot_iv_surface(iv_data, bates_model, S0_market, r_market)
                else:
                    print("Could not calculate implied volatility for any options.")

    except (ValueError, RuntimeError, Exception) as e:
        print(f"\nAn error occurred during execution: {e}")
        print("The process could not be completed.")
