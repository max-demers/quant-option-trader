import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.optimize import differential_evolution
import yfinance as yf
import datetime


# --- MODEL AND PRICING FUNCTIONS ---

def heston_pricer_robust(S0, K, T, r, v0, kappa, theta, rho, sigma):
    """
    (ROBUST AND VECTORIZED) Calculates Heston analytical call price for multiple strikes.
    This implementation is designed for numerical stability.
    K can be a single value or a numpy array of strikes.
    """
    K = np.atleast_1d(K)
    num_phi_points = 2000
    phi_max = 50.0
    phi = np.linspace(1e-6, phi_max, num_phi_points)

    # --- P1 and P2 Calculation ---
    P = np.zeros((2, len(K)))  # Array to store P1 and P2 results

    for j in [1, 2]:
        if j == 1:
            u, b = 0.5, kappa - rho * sigma
        else:  # j == 2
            u, b = -0.5, kappa

        a = kappa * theta
        with np.errstate(all='ignore'):  # Suppress warnings during optimization
            d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 + sigma ** 2 * (phi ** 2 - 2 * u * 1j * phi))
            g = (b - rho * sigma * 1j * phi - d) / (b - rho * sigma * 1j * phi + d)
            exp_neg_dT = np.exp(-d * T)
            # This is a more stable form of the characteristic function's C and D components
            C = r * 1j * phi * T + (a / sigma ** 2) * (
                        (b - rho * sigma * 1j * phi - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1 - g)))
            D = ((b - rho * sigma * 1j * phi - d) / sigma ** 2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT))

        char_func = np.exp(C + D * v0 + 1j * phi * np.log(S0))

        # Vectorized integrand calculation
        z_integrand_numerator = np.exp(-1j * phi[np.newaxis, :] * np.log(K[:, np.newaxis])) * char_func[np.newaxis, :]
        integrand = np.imag(z_integrand_numerator) / phi[np.newaxis, :]

        # Graph of heston Integrand for P1 and P2
        """plt.figure(figsize=(10,5))
        plt.plot(phi, integrand[0,:]),
        label = "Heston Integrand for P"
        color ='blue'

        plt.title("Visualisation de la convergence de l'intégrale de Fourier-Heston")
        plt.xlabel("Fréquence(Phi)")
        plt.ylabel("Valeur de l'intégrande")
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.grid(True, alpha=0.3)
        plt.show()"""

        # Integration
        integral = simpson(integrand, x=phi, axis=1)
        P[j - 1, :] = 0.5 + (1 / np.pi) * integral

    # Final price calculation
    price = S0 * P[0, :] - K * np.exp(-r * T) * P[1, :]

    # Return a single value if only one strike was passed
    if len(price) == 1:
        return price[0]
    return price


def objective_function(params, market_data, S0, r):
    """
    Calculates the sum of squared errors across multiple expiries.
    market_data is a DataFrame with columns ['T', 'strike', 'price']
    """
    v0, kappa, theta, rho, sigma = params

    # The market_data DataFrame should be sorted by 'T' to ensure order.
    all_model_prices = []

    try:
        # Group by time to expiry T
        for T, group in market_data.groupby('T'):
            strikes = group['strike'].values

            # Calculate model prices for this expiry
            model_prices_T = heston_pricer_robust(S0, strikes, T, r, v0, kappa, theta, rho, sigma)

            # Ensure we always have an array, even if only one strike
            model_prices_T = np.atleast_1d(model_prices_T)

            # Handle potential errors from the pricer for this group
            if np.isnan(model_prices_T).any():
                return 1e10  # Return a large error if any price is NaN

            all_model_prices.append(model_prices_T)

        # Concatenate prices from all expiries. The order is guaranteed by groupby, which sorts keys.
        model_prices = np.concatenate(all_model_prices)

        market_prices = market_data['price'].values

        error = np.sum((model_prices - market_prices) ** 2)

        # Final check, although the loop should have caught it.
        if np.isnan(error):
            return 1e10

    except Exception:
        # If any other error occurs during pricing (e.g., from bad params), return a large value.
        return 1e10

    return error


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly

     # --- Part 2: Calibration to Market Data ---
    print("\n--- Part 2: Calibration to Market Data ---")

    # Initialize variables that will be populated with market or fallback data
    market_prices, strikes, expiration, T_market, S0_market, r_market = None, None, None, None, None, None

    market_data = None  # This will become a DataFrame
    try:
        # --- Attempt to fetch live market data for specific target expiries ---
        print("Searching for specific expiry targets (1w, 2w, ~1.5m, 1y)...")
        ticker = yf.Ticker("AAPL")
        all_expirations = ticker.options

        # 1. Pre-process all available expirations
        today = datetime.date.today()
        expiries_with_days = []
        for exp_date_str in all_expirations:
            expiration_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
            days = (expiration_date - today).days
            if days > 0:  # Only consider future dates
                expiries_with_days.append({'date_str': exp_date_str, 'days': days})

        # 2. Define targets and find best match for each
        targets_in_days = [7, 14, 45, 365]
        selected_expiries = set()  # To store date strings we've already selected
        market_data_frames = []

        # Get S0 and strike bounds once
        S0_market = ticker.history(period="1d")['Close'].iloc[-1]
        lower_strike_bound, upper_strike_bound = S0_market * 0.8, S0_market * 1.2
        MIN_OPTIONS_PER_EXPIRY = 8
        MIN_VOLUME = 10

        for target_days in targets_in_days:
            # Sort potential expiries by their proximity to the current target
            potential_matches = []
            for expiry in expiries_with_days:
                if expiry['date_str'] in selected_expiries:
                    continue
                diff = abs(expiry['days'] - target_days)
                potential_matches.append({'expiry': expiry, 'diff': diff})

            potential_matches.sort(key=lambda x: x['diff'])

            found_liquid_expiry_for_target = False
            for match in potential_matches:
                best_match = match['expiry']
                exp_date_str = best_match['date_str']
                days_to_expiry = best_match['days']

                # Check if this potential match is suitable
                chain = ticker.option_chain(exp_date_str)
                calls = chain.calls

                valid_market = (
                        (calls['volume'] > MIN_VOLUME) & (calls['bid'] > 0) &
                        (calls['strike'] >= lower_strike_bound) & (calls['strike'] <= upper_strike_bound)
                )

                if valid_market.sum() >= MIN_OPTIONS_PER_EXPIRY:
                    print(
                        f"Found suitable liquid match for target ~{target_days} days: {exp_date_str} ({days_to_expiry} days) with {valid_market.sum()} options.")

                    expiry_data = calls[valid_market].copy()
                    expiry_data['price'] = (expiry_data['bid'] + expiry_data['ask']) / 2
                    expiry_data['T'] = days_to_expiry / 365.25

                    market_data_frames.append(expiry_data[['T', 'strike', 'price']])
                    selected_expiries.add(exp_date_str)  # Mark as used
                    found_liquid_expiry_for_target = True
                    break  # Exit the inner loop and move to the next target

            if not found_liquid_expiry_for_target:
                print(f"Could not find any liquid expiry for target ~{target_days} days. Skipping this target.")

        if market_data_frames:
            market_data = pd.concat(market_data_frames, ignore_index=True).sort_values(by='T').reset_index(drop=True)
            r_market = 0.05  # Use theoretical risk-free rate for now

        if market_data is None or market_data.empty:
            raise ValueError(f"Could not find enough liquid options for the specified targets.")

    except Exception as e:
        # --- Fallback to hardcoded data if fetching fails ---
        print(f"\nCould not fetch live market data: {e}")
        print("Using hardcoded fallback data for testing.")

        S0_market = 180.0
        r_market = 0.05
        # Create a multi-expiry DataFrame for fallback
        data_t1 = {'T': 0.25, 'strike': [165, 170, 175, 180, 185, 190, 195],
                   'price': [18.05, 14.50, 11.20, 8.45, 6.10, 4.25, 2.80]}
        data_t2 = {'T': 0.50, 'strike': [165, 170, 175, 180, 185, 190, 195],
                   'price': [22.80, 19.80, 17.00, 14.40, 12.10, 10.05, 8.30]}  # Fictional prices
        market_data = pd.concat([pd.DataFrame(data_t1), pd.DataFrame(data_t2)], ignore_index=True)

    # --- Run Calibration if data (live or fallback) is available ---
    if market_data is not None and not market_data.empty:
        print(f"\nCalibrating to {len(market_data)} options across {market_data['T'].nunique()} maturities.")
        print(f"Underlying Price (S0): {S0_market:.2f}")

        bounds = [(0.001, 0.9), (0.2, 8.0), (0.01, 0.5), (-0.9, -0.3), (0.1, 2.0)]

        print("\nRunning optimization with differential_evolution... This may take longer but is more robust.")
        result = differential_evolution(
            objective_function, bounds,
            args=(market_data, S0_market, r_market),
            maxiter=100, popsize=15, tol=0.01, updating='deferred', workers=-1
        )

        if result.success:
            v0_opt, kappa_opt, theta_opt, rho_opt, sigma_opt = result.x
            rmse = np.sqrt(result.fun / len(market_data))
            print("\nCalibration Successful!")
            print(
                f"Calibrated Parameters: v0={v0_opt:.4f}, kappa={kappa_opt:.4f}, theta={theta_opt:.4f}, rho={rho_opt:.4f}, sigma={sigma_opt:.4f}")
            print(f"Final Minimized Error (Sum of Squares): {result.fun:.4f}")
            print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

            # --- Part A: Visualize Calibration Fit on Initial Dataset ---
            print("\nGenerating comparison plot for the calibration data...")
            all_model_prices_opt = []
            for T, group in market_data.groupby('T'):
                strikes = group['strike'].values
                model_prices_T = heston_pricer_robust(S0_market, strikes, T, r_market, v0_opt, kappa_opt, theta_opt,
                                                      rho_opt, sigma_opt)
                all_model_prices_opt.append(pd.Series(model_prices_T, index=group.index))
            market_data['model_price'] = pd.concat(all_model_prices_opt)

            # Create a plot for each expiry date used in calibration
            unique_expiries_calib = sorted(market_data['T'].unique())
            num_expiries_calib = len(unique_expiries_calib)

            fig, axes = plt.subplots(1, num_expiries_calib, figsize=(6 * num_expiries_calib, 5), squeeze=False)
            fig.suptitle('Market Prices vs. Calibrated Heston Model (Calibration Set)', fontsize=16)

            for i, T_val in enumerate(unique_expiries_calib):
                ax = axes[0, i]
                expiry_data = market_data[market_data['T'] == T_val].sort_values(by='strike')
                ax.plot(expiry_data['strike'], expiry_data['price'], 'o-', label='Market Prices')
                ax.plot(expiry_data['strike'], expiry_data['model_price'], 'x--', label='Heston Model')
                ax.set_title(f'Expiry T = {T_val:.3f} years (~{int(T_val * 365.25)} days)')
                ax.set_xlabel('Strike Price (K)')
                ax.set_ylabel('Option Price')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # --- Part B: Full Market Scan for Under/Over-valued Options ---
            print("\n--- Full Market Scan for Trading Opportunities ---")
            print("Fetching all available option chains (expiry >= 7 days) to find discrepancies...")

            all_expiries_full = ticker.options
            full_market_data_frames = []
            MIN_DAYS_SCAN = 7
            MIN_OPEN_INTEREST_SCAN = 100
            print(f"Applying additional scan filters: Open Interest >= {MIN_OPEN_INTEREST_SCAN}, Strike within +/- 20% of S0.")
            # Using the same MIN_VOLUME from calibration for consistency

            for exp_date_str in all_expiries_full:
                expiration_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                days_to_expiry = (expiration_date - today).days

                if days_to_expiry >= MIN_DAYS_SCAN:
                    try:
                        chain = ticker.option_chain(exp_date_str)
                        calls = chain.calls.copy()  # Use a copy to avoid SettingWithCopyWarning

                        # Filter for liquid options with high open interest and relevant strikes
                        valid_market_scan = (
                                (calls['volume'] > MIN_VOLUME) &
                                (calls['openInterest'] >= MIN_OPEN_INTEREST_SCAN) &
                                (calls['bid'] > 0) &
                                (calls['ask'] > 0) &
                                (calls['strike'] >= lower_strike_bound) &
                                (calls['strike'] <= upper_strike_bound)
                        )

                        if valid_market_scan.sum() > 0:
                            print(f"  Scanning expiry {exp_date_str} ({days_to_expiry} days), found {valid_market_scan.sum()} liquid options.")

                            expiry_data_scan = calls[valid_market_scan].copy()
                            expiry_data_scan['price'] = (expiry_data_scan['bid'] + expiry_data_scan['ask']) / 2
                            expiry_data_scan['T'] = days_to_expiry / 365.25

                            # Keep relevant columns plus 'volume' and 'openInterest' for context
                            full_market_data_frames.append(
                                expiry_data_scan[['T', 'strike', 'price', 'volume', 'openInterest']])

                    except Exception as e:
                        print(f"  Could not process expiry {exp_date_str}. Error: {e}")

            if full_market_data_frames:
                market_data_full_scan = pd.concat(full_market_data_frames, ignore_index=True)
                print(f"\nScan complete. Found a total of {len(market_data_full_scan)} liquid options for analysis.")

                print("Pricing all found options with the calibrated Heston parameters...")
                all_model_prices_scan = []
                # Grouping by T is efficient for the pricer
                for T_scan, group in market_data_full_scan.groupby('T'):
                    strikes_scan = group['strike'].values
                    model_prices_T_scan = heston_pricer_robust(S0_market, strikes_scan, T_scan, r_market, v0_opt,
                                                               kappa_opt, theta_opt, rho_opt, sigma_opt)
                    all_model_prices_scan.append(pd.Series(model_prices_T_scan, index=group.index))

                market_data_full_scan['model_price'] = pd.concat(all_model_prices_scan)

                # Calculate difference (not absolute) to see direction
                market_data_full_scan['price_diff'] = market_data_full_scan['model_price'] - \
                                                      market_data_full_scan['price']

                # Sort to find most overvalued (model >> market)
                overvalued_options = market_data_full_scan.sort_values(by='price_diff', ascending=False).head(10)

                # Sort to find most undervalued (model << market)
                undervalued_options = market_data_full_scan.sort_values(by='price_diff', ascending=True).head(10)

                print("\n--- Top 10 Potentially OVERVALUED Calls (Model Price > Market Price) ---")
                print(overvalued_options[['T', 'strike', 'price', 'model_price', 'price_diff', 'volume',
                                          'openInterest']].round(4))

                print("\n--- Top 10 Potentially UNDERVALUED Calls (Model Price < Market Price) ---")
                print(undervalued_options[['T', 'strike', 'price', 'model_price', 'price_diff', 'volume',
                                         'openInterest']].round(4))

            else:
                print("\nScan complete. No liquid options found meeting the criteria for the full market scan.")



        else:
            print("\nCalibration failed.")
            print(f"Reason: {result.message}")
    else:
        print("\nNo data available to run calibration.")
