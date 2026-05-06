import yfinance as yf
import pandas as pd
import datetime
import re

class MarketDataHandler:
    """
    Handles all interactions with the market data provider (yfinance).
    """
    def __init__(self, ticker_symbol):
        """
        Initializes the data handler for a specific ticker.

        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        """
        self.ticker = yf.Ticker(ticker_symbol)
        self.ticker_symbol = ticker_symbol
        print(f"MarketDataHandler initialized for ticker: {self.ticker_symbol}")

    def get_current_price(self):
        """
        Fetches the most recent price for the stock.
        Tries to get a near real-time price from 1-minute interval data,
        and falls back to the last daily close if unavailable.
        """
        try:
            # Fetch 1-minute interval data for the last 2 days to ensure data is available.
            # The last point in the series will be the most recent price.
            data = self.ticker.history(period="2d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                # If 1m data is empty (e.g., pre-market), fall back to daily.
                print("Intraday (1m) data is empty, falling back to daily history.")
                return self.ticker.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            # This can happen if the market is closed and yfinance returns no intraday data.
            print(f"Could not fetch intraday (1m) data: {e}. Falling back to daily history.")
            # Fallback to the previous, more robust method
            return self.ticker.history(period="1d")['Close'].iloc[-1]

    def get_risk_free_rate(self):
        """
        Fetches a proxy for the risk-free rate.
        Using the 13-week Treasury Bill yield ('^IRX').
        In a real application, you might use a more stable source or a direct API.
        """
        try:
            # Fetch data for the last few days and take the most recent close
            irx = yf.Ticker("^IRX")
            rate = irx.history(period="5d")['Close'].iloc[-1]
            return rate / 100  # Convert percentage to decimal
        except Exception as e:
            print(f"Could not fetch live risk-free rate. Error: {e}. Falling back to 0.05.")
            return 0.0225 # Fallback value

    def fetch_calibration_data(self, S0, targets_in_days, strike_window_pct, min_options, min_volume):
        """
        Fetches and filters market data for model calibration, targeting specific expiries.

        Args:
            S0 (float): Current underlying price.
            targets_in_days (list): A list of target expiries in days (e.g., [7, 14, 45, 365]).
            strike_window_pct (float): The percentage window around S0 for strikes (e.g., 0.08 for +/- 8%).
            min_options (int): Minimum number of liquid options required per expiry.
            min_volume (int): Minimum volume for an option to be considered liquid.

        Returns:
            A pandas DataFrame with filtered market data for calibration, or None.
        """
        print("\nSearching for specific expiry targets for calibration...")
        all_expirations = self.ticker.options
        today = datetime.date.today()
        
        expiries_with_days = []
        for exp_date_str in all_expirations:
            expiration_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
            days = (expiration_date - today).days
            if days >= 5: # Only consider expiries at least 5 days out
                expiries_with_days.append({'date_str': exp_date_str, 'days': days})

        selected_expiries = set()
        market_data_frames = []
        lower_strike_bound, upper_strike_bound = S0 * (1 - strike_window_pct), S0 * (1 + strike_window_pct)

        for target_days in targets_in_days:
            # Sort potential expiries by their proximity to the current target
            potential_matches = sorted(
                [exp for exp in expiries_with_days if exp['date_str'] not in selected_expiries],
                key=lambda x: abs(x['days'] - target_days)
            )

            found_liquid_expiry = False
            for match in potential_matches:
                exp_date_str = match['date_str']
                days_to_expiry = match['days']

                chain = self.ticker.option_chain(exp_date_str)
                calls = chain.calls

                valid_market = (
                    (calls['volume'] > min_volume) &
                    (calls['bid'] > 0) &
                    (calls['strike'] >= lower_strike_bound) &
                    (calls['strike'] <= upper_strike_bound)
                )

                if valid_market.sum() >= min_options:
                    print(f"  Found suitable match for target ~{target_days} days: {exp_date_str} ({days_to_expiry} days) with {valid_market.sum()} options.")
                    expiry_data = calls[valid_market].copy()
                    expiry_data['price'] = (expiry_data['bid'] + expiry_data['ask']) / 2
                    expiry_data['T'] = days_to_expiry / 365.25
                    market_data_frames.append(expiry_data[['T', 'strike', 'price']])
                    selected_expiries.add(exp_date_str)
                    found_liquid_expiry = True
                    break
            
            if not found_liquid_expiry:
                print(f"  Could not find a liquid expiry for target ~{target_days} days.")

        if market_data_frames:
            market_data = pd.concat(market_data_frames, ignore_index=True).sort_values(by='T').reset_index(drop=True)
            return market_data
        else:
            print("Could not find enough liquid options for calibration.")
            return None

    def fetch_all_tradable_options(self, S0, config):
        """
        Fetches all options that meet the criteria for a full market scan.

        Args:
            S0 (float): Current underlying price.
            config (dict): A dictionary with scanning parameters.
        
        Returns:
            A pandas DataFrame with all filtered tradable options.
        """
        print("\nFetching all available option chains for trading scan...")
        all_expiries_full = self.ticker.options
        today = datetime.date.today()
        full_market_data_frames = []

        lower_strike_bound = S0 * (1 - config['strike_window_pct'])
        upper_strike_bound = S0 * (1 + config['strike_window_pct'])

        for exp_date_str in all_expiries_full:
            expiration_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
            days_to_expiry = (expiration_date - today).days

            if config['min_days_to_expiry'] <= days_to_expiry <= config['max_days_to_expiry']:
                try:
                    chain = self.ticker.option_chain(exp_date_str)
                    calls = chain.calls.copy()

                    valid_market_scan = (
                        (calls['volume'] > config['min_volume']) &
                        (calls['openInterest'] >= config['min_open_interest']) &
                        (calls['bid'] > 0) &
                        (calls['ask'] > 0) &
                        (calls['strike'] >= lower_strike_bound) &
                        (calls['strike'] <= upper_strike_bound)
                    )

                    if valid_market_scan.sum() > 0:
                        print(f"  Scanning expiry {exp_date_str} ({days_to_expiry} days), found {valid_market_scan.sum()} liquid options.")
                        
                        expiry_data_scan = calls[valid_market_scan].copy()
                        # Use contractSymbol as a unique ID for positions
                        expiry_data_scan.rename(columns={'contractSymbol': 'contract_id'}, inplace=True)
                        expiry_data_scan['price'] = (expiry_data_scan['bid'] + expiry_data_scan['ask']) / 2
                        expiry_data_scan['T'] = days_to_expiry / 365.25
                        
                        full_market_data_frames.append(expiry_data_scan)

                except Exception as e:
                    print(f"  Could not process expiry {exp_date_str}. Error: {e}")

        if full_market_data_frames:
            market_data_full_scan = pd.concat(full_market_data_frames, ignore_index=True)
            print(f"\nScan complete. Found a total of {len(market_data_full_scan)} liquid options for analysis.")
            return market_data_full_scan
        else:
            print("\nScan complete. No liquid options found meeting the criteria.")
            return None

    def fetch_data_for_portfolio_positions(self, contract_ids: list):
        """
        Fetches market data specifically for a list of option contract IDs.
        This is more efficient than a full market scan when only checking existing positions.
        """
        if not contract_ids:
            return pd.DataFrame()

        print("\nFetching market data for specific portfolio positions...")

        # Helper to parse yfinance contract ID format
        def parse_expiry_from_id(contract_id):
            match = re.search(r'\D(\d{6})', contract_id)
            if not match:
                return None
            date_str = match.group(1)
            try:
                # Assuming format YYMMDD
                expiry_date = datetime.datetime.strptime(date_str, "%y%m%d").date()
                return expiry_date.strftime("%Y-%m-%d")
            except ValueError:
                return None

        # Group contracts by expiry to minimize API calls
        expiries_to_fetch = {parse_expiry_from_id(cid) for cid in contract_ids}
        expiries_to_fetch.discard(None)  # Remove any IDs that failed to parse

        if not expiries_to_fetch:
            print("Could not parse expiry dates from any of the provided contract IDs.")
            return pd.DataFrame()

        all_found_options = []
        for exp_date_str in expiries_to_fetch:
            try:
                chain = self.ticker.option_chain(exp_date_str)
                # Combine calls and puts to ensure we find the contract regardless of type
                options_for_expiry = pd.concat([chain.calls, chain.puts], ignore_index=True)
                all_found_options.append(options_for_expiry)
                print(f"  Fetched option chain for expiry {exp_date_str}.")
            except Exception as e:
                print(f"  Could not fetch option chain for expiry {exp_date_str}. Error: {e}")

        if not all_found_options:
            print("Failed to fetch data for any of the required expiry dates.")
            return pd.DataFrame()

        # Combine all chains and then filter for the specific contracts we need
        market_data = pd.concat(all_found_options, ignore_index=True)
        market_data.rename(columns={'contractSymbol': 'contract_id'}, inplace=True)

        portfolio_data = market_data[market_data['contract_id'].isin(contract_ids)].copy()

        if portfolio_data.empty:
            print("Found expiry chains, but none contained the specific contract IDs.")
            return pd.DataFrame()

        # --- Robustly add date-related columns ---
        today = datetime.date.today()
        today_dt = pd.to_datetime(today)  # Convert once for vectorized operation

        # Extract date strings from contract ID
        date_strings = portfolio_data['contract_id'].str.extract(r'\D(\d{6})')[0]

        # Convert to datetime, coercing errors to NaT (Not a Time)
        expiration_dates = pd.to_datetime(date_strings, format='%y%m%d', errors='coerce')

        # Calculate timedelta. The result will be NaT for any rows that failed conversion.
        time_deltas = expiration_dates - today_dt

        # Now, we can safely use the .dt accessor because the series is guaranteed to be a timedelta type
        portfolio_data['days_to_expiry'] = time_deltas.dt.days
        portfolio_data['T'] = portfolio_data['days_to_expiry'] / 365.25
        portfolio_data['price'] = (portfolio_data['bid'] + portfolio_data['ask']) / 2

        # Drop any rows where we failed to calculate the expiry date and clean up dtypes
        portfolio_data.dropna(subset=['days_to_expiry'], inplace=True)
        portfolio_data['days_to_expiry'] = portfolio_data['days_to_expiry'].astype(int)

        return portfolio_data
