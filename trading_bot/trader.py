import schedule
import time
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# Import project modules
import Project.trading_bot.config as config
from Project.trading_bot.market_data_handler import MarketDataHandler
from Project.trading_bot.bates_model import BatesModel
from Project.trading_bot.portfolio_manager import Portfolio

class TradingBot:
    """
    A class that encapsulates the entire trading bot logic.
    This structure prevents re-initialization issues when using multiprocessing.
    """
    def __init__(self):
        """
        Initializes all necessary components for the bot.
        """
        self.market_handler = MarketDataHandler(config.TICKER_SYMBOL)
        self.bates_model = BatesModel()
        self.portfolio = Portfolio(config.INITIAL_CASH)
        
        print("--- Trader Application Starting ---")
        print(f"Ticker: {config.TICKER_SYMBOL}")
        print("------------------------------------")

    def is_market_open(self):
        """
        Checks if the NASDAQ market is currently open.
        """
        nasdaq = mcal.get_calendar('NASDAQ')
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        schedule = nasdaq.schedule(start_date=today_str, end_date=today_str)

        if schedule.empty:
            return False

        now_utc = datetime.datetime.now(timezone.utc)
        now_market_tz = now_utc.astimezone(schedule.index[0].tz)
        
        market_open = schedule.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule.iloc[0]['market_close'].to_pydatetime()

        return market_open <= now_market_tz <= market_close

    def run_calibration(self):
        """
        Fetches fresh market data and recalibrates the Bates model.
        (This version is for LIVE mode only)
        """
        print(f"\n>>> [{datetime.datetime.now()}] RUNNING CALIBRATION TASK <<<")
        try:
            S0 = self.market_handler.get_current_price()
            r = self.market_handler.get_risk_free_rate()
            calibration_data = self.market_handler.fetch_calibration_data(
                S0=S0, targets_in_days=config.CALIBRATION_TARGETS_DAYS, 
                strike_window_pct=config.CALIBRATION_STRIKE_WINDOW_PCT,
                min_options=config.CALIBRATION_MIN_OPTIONS_PER_EXPIRY, 
                min_volume=config.CALIBRATION_MIN_VOLUME
            )

            if calibration_data is None or calibration_data.empty:
                print("Calibration skipped: Not enough data.")
                return

            self.bates_model.calibrate(calibration_data, S0, r, config.OPTIMIZER_BOUNDS, config.OPTIMIZER_SETTINGS)
            print(">>> CALIBRATION TASK COMPLETE <<<")

        except Exception as e:
            print(f"!!! ERROR during calibration task: {e} !!!")

    def run_trading_scan(self):
        """
        Scans for trading opportunities and evaluates existing positions for closure.
        This function is now more robust, ensuring held positions are always evaluated
        by fetching their data specifically, even if they become illiquid.
        """
        now = datetime.datetime.now()
        if not self.is_market_open():
            print(f"[{now}] Market is closed. Skipping trading scan.")
            return

        print(f"\n>>> [{now}] RUNNING TRADING SCAN <<<")

        if self.bates_model.rmse is None:
            print("Trading scan skipped: Model is not calibrated yet.")
            return

        try:
            S0 = self.market_handler.get_current_price()
            r = self.market_handler.get_risk_free_rate()

            # --- Step 1: Fetch data for all currently held positions ---
            held_contract_ids = list(self.portfolio.positions.keys())
            portfolio_options_data = self.market_handler.fetch_data_for_portfolio_positions(held_contract_ids)

            # --- Step 2: Fetch all generally tradable options for new opportunities ---
            tradable_options_for_scan = self.market_handler.fetch_all_tradable_options(S0, config.SCAN_CONFIG)

            # --- Step 3: Combine datasets for a complete market view ---
            # This ensures we have data for held positions even if they are no longer "tradable"
            if tradable_options_for_scan is None or tradable_options_for_scan.empty:
                all_market_options = portfolio_options_data.copy() if not portfolio_options_data.empty else pd.DataFrame()
            elif portfolio_options_data.empty:
                all_market_options = tradable_options_for_scan.copy()
            else:
                all_market_options = pd.concat([portfolio_options_data, tradable_options_for_scan]).drop_duplicates(
                    subset=['contract_id']).reset_index(drop=True)

            if all_market_options.empty:
                print("No options found for analysis (neither held nor generally tradable).")
                self.portfolio.get_pnl(S0, None)
                return

            # --- Step 4: Price all relevant options with the model ---
            model_prices, deltas = [], []
            for T_scan, group in all_market_options.groupby('T'):
                prices, d = self.bates_model.bates_pricer_robust(S0, group['strike'].values, T_scan, r)
                model_prices.append(pd.Series(prices, index=group.index))
                deltas.append(pd.Series(d, index=group.index))
            all_market_options['model_price'] = pd.concat(model_prices)
            all_market_options['delta'] = pd.concat(deltas)
            all_market_options.dropna(subset=['model_price', 'delta'], inplace=True) # Drop rows where pricing failed

            # --- Step 5: Evaluate existing positions for closure ---
            for contract_id in held_contract_ids:
                if contract_id not in self.portfolio.positions:
                    continue  # Already closed in this loop

                if contract_id not in all_market_options['contract_id'].values:
                    print(f"INFO: Position {contract_id} not found in market data (likely delisted). Cannot evaluate for closure.")
                    continue

                current_option_data = all_market_options.loc[all_market_options['contract_id'] == contract_id].iloc[0]

                # Rule 1: Close if nearing expiry (7 days)
                days_to_expiry = current_option_data['T'] * 365.25
                if days_to_expiry <= 7:
                    print(f"INFO: Closing {contract_id} due to 7-day expiry rule (DTE: {days_to_expiry:.1f}).")
                    position_details = self.portfolio.positions[contract_id]
                    quantity_to_close = -position_details['quantity']
                    price_to_transact = current_option_data['bid'] if quantity_to_close < 0 else current_option_data['ask']

                    if price_to_transact > 0:
                        self.portfolio.transact_option(contract_id, quantity_to_close, price_to_transact, now)
                        # Unwind hedge if successful
                        if contract_id not in self.portfolio.positions:
                            delta_of_trade = current_option_data['delta']
                            shares_to_hedge = -1 * quantity_to_close * delta_of_trade * 100
                            self.portfolio.update_stock_position(config.TICKER_SYMBOL, int(round(shares_to_hedge)), S0, now)
                    else:
                        print(f"WARNING: Invalid transaction price ({price_to_transact}) for closing {contract_id} on 7-day rule. Skipping.")
                    continue # Move to next position

                # Rule 2: Close if price has converged
                market_mid_price = current_option_data['price']
                model_price = current_option_data['model_price']
                if abs(market_mid_price - model_price) / model_price < config.CLOSE_POSITION_THRESHOLD_PCT:
                    print(f"INFO: Closing position {contract_id} as prices converged.")
                    position_details = self.portfolio.positions[contract_id]
                    quantity_to_close = -position_details['quantity']
                    price_to_transact = current_option_data['bid'] if quantity_to_close < 0 else current_option_data['ask']

                    if price_to_transact > 0:
                        self.portfolio.transact_option(contract_id, quantity_to_close, price_to_transact, now)
                        # Unwind hedge if successful
                        if contract_id not in self.portfolio.positions:
                            delta_of_trade = current_option_data['delta']
                            shares_to_hedge = -1 * quantity_to_close * delta_of_trade * 100
                            self.portfolio.update_stock_position(config.TICKER_SYMBOL, int(round(shares_to_hedge)), S0, now)
                    else:
                        print(f"WARNING: Invalid transaction price ({price_to_transact}) for closing {contract_id}. Skipping.")


            # --- Step 6: Find and execute new opportunities ---
            # Candidates are options that are generally tradable and not already in the portfolio
            if tradable_options_for_scan is not None and not tradable_options_for_scan.empty:
                new_opp_candidates = all_market_options[
                    all_market_options['contract_id'].isin(tradable_options_for_scan['contract_id']) &
                    ~all_market_options['contract_id'].isin(list(self.portfolio.positions.keys()))
                ].copy()

                if not new_opp_candidates.empty:
                    new_opp_candidates['price_diff'] = new_opp_candidates['model_price'] - new_opp_candidates['price']
                    new_opp_candidates['spread_cost'] = (new_opp_candidates['ask'] - new_opp_candidates['bid']) / 2
                    new_opp_candidates['theo_profit'] = np.abs(new_opp_candidates['price_diff']) - new_opp_candidates['spread_cost']
                    outlier_threshold = config.PROFIT_THRESHOLD_MULTIPLIER * self.bates_model.rmse
                    opportunities = new_opp_candidates[new_opp_candidates['theo_profit'] > outlier_threshold]

                    if not opportunities.empty:
                        best_opportunity = opportunities.sort_values(by='theo_profit', ascending=False).iloc[0]
                        price_diff = best_opportunity['price_diff']
                        quantity_to_open = 1 if price_diff > 0 else -1
                        price_to_transact = best_opportunity['ask'] if quantity_to_open > 0 else best_opportunity['bid']

                        if price_to_transact > 0:
                            print(f"INFO: Found new opportunity. {'Buying' if quantity_to_open > 0 else 'Shorting'} {best_opportunity['contract_id']} at ${price_to_transact:.2f}")
                            self.portfolio.transact_option(best_opportunity['contract_id'], quantity_to_open, price_to_transact, now)
                            delta_of_trade = best_opportunity['delta']
                            shares_to_hedge = -1 * quantity_to_open * delta_of_trade * 100
                            print(f"INFO: Hedging new trade, requires {int(round(-shares_to_hedge))} shares.")
                            self.portfolio.update_stock_position(config.TICKER_SYMBOL, int(round(shares_to_hedge)), S0, now)
                        else:
                            print(f"WARNING: Invalid transaction price ({price_to_transact}) for opening {best_opportunity['contract_id']}. Skipping.")
                    else:
                        print("INFO: No significant new opportunities found. Market appears balanced.")
                else:
                    print("INFO: No new opportunities found. All tradable options may already be in portfolio.")
            else:
                print("INFO: No generally tradable options found to scan for new opportunities.")


            self.portfolio.get_pnl(S0, all_market_options)
            print(">>> TRADING SCAN COMPLETE <<<")

        except Exception as e:
            print(f"!!! ERROR during trading scan: {e} !!!")

    def run_hedge_adjustment(self):
        """
        Reconciles portfolio delta based on current positions.
        This function now implements a more efficient check:
        1. Fetches data ONLY for held positions.
        2. Checks each position for liquidity or upcoming expiry.
        3. Liquidates if necessary.
        4. Calculates total delta of remaining valid positions and adjusts the hedge.
        """
        now = datetime.datetime.now()
        if not self.is_market_open():
            print(f"[{now}] Market is closed. Skipping hedge reconciliation.")
            return

        print(f"\n>>> [{now}] RUNNING EFFICIENT HEDGE RECONCILIATION <<<")

        if not self.portfolio.positions:
            print("No open positions to hedge. Skipping.")
            # We can still update PnL with the current stock price
            try:
                S0_for_pnl = self.market_handler.get_current_price()
                self.portfolio.get_pnl(S0_for_pnl, None)
            except Exception as e:
                print(f"Could not fetch price for PnL update: {e}")
            return

        try:
            S0 = self.market_handler.get_current_price()
            r = self.market_handler.get_risk_free_rate()

            held_contract_ids = list(self.portfolio.positions.keys())

            # Step 1: Fetch data only for the options we hold
            portfolio_options_data = self.market_handler.fetch_data_for_portfolio_positions(held_contract_ids)

            # Create a set of contract IDs that were successfully found for quick lookups
            found_contract_ids = set(portfolio_options_data['contract_id']) if not portfolio_options_data.empty else set()

            total_portfolio_delta = 0

            # Iterate over all held positions
            for contract_id in held_contract_ids:
                # This check must be inside the loop because a position might be liquidated
                if contract_id not in self.portfolio.positions:
                    continue

                position_details = self.portfolio.positions[contract_id]

                # Check if we found market data for this specific option
                if contract_id not in found_contract_ids:
                    # This option is illiquid or delisted. We have no price, so we can't sell.
                    print(f"CRITICAL: Position {contract_id} is illiquid or delisted (no market data found). Cannot sell or hedge. Manual review needed.")
                    continue

                # If we are here, we have market data for the option
                option_data = portfolio_options_data[portfolio_options_data['contract_id'] == contract_id].iloc[0]

                # Step 2: Check conditions for liquidation
                days_to_expiry = option_data['days_to_expiry']
                is_illiquid = option_data['volume'] < config.SCAN_CONFIG['min_volume'] or option_data['bid'] <= 0

                if is_illiquid or days_to_expiry < 10:
                    reason = "illiquid" if is_illiquid else f"expires in {days_to_expiry} days (<10)"
                    print(f"INFO: Liquidating {contract_id} because it is {reason}.")

                    quantity_to_close = -position_details['quantity']
                    price_to_transact = option_data['bid'] if position_details['quantity'] > 0 else option_data['ask']

                    if price_to_transact > 0:
                        self.portfolio.transact_option(contract_id, quantity_to_close, price_to_transact, now)
                    else:
                        print(f"WARNING: Could not liquidate {contract_id} due to invalid transaction price ({price_to_transact}).")

                    continue  # Move to the next position

                # Step 3: If not liquidated, calculate its delta for the hedge
                _, delta_result = self.bates_model.bates_pricer_robust(S0, [option_data['strike']], option_data['T'], r)

                if delta_result is None or not delta_result or delta_result[0] is None:
                    print(f"CRITICAL: Delta calculation failed for {contract_id}. Position will not be hedged this cycle.")
                    continue

                delta = delta_result[0]
                position_delta = delta * position_details['quantity'] * 100
                total_portfolio_delta += position_delta

            # Step 4: Adjust the hedge based on the sum of deltas of valid positions
            print("\n--- Hedge Adjustment ---")
            current_stock_hedge = self.portfolio.stock_holdings.get(config.TICKER_SYMBOL, 0)
            required_stock_position = -total_portfolio_delta
            shares_to_trade = int(round(required_stock_position - current_stock_hedge))

            if abs(shares_to_trade) > 0:
                print(f"Total Portfolio Option Delta: {total_portfolio_delta:+.2f}")
                print(f"Current Stock Hedge: {current_stock_hedge} shares")
                print(f"Net Delta: {(total_portfolio_delta + current_stock_hedge):+.2f}")
                print(f"Required change: {shares_to_trade} shares")
                self.portfolio.update_stock_position(config.TICKER_SYMBOL, shares_to_trade, S0, now)
            else:
                print("No reconciliation needed. Hedge is stable.")

            self.portfolio.get_pnl(S0, portfolio_options_data)
            print(">>> HEDGE RECONCILIATION COMPLETE <<<")

        except Exception as e:
            print(f"!!! ERROR during hedge reconciliation: {e} !!!")

    def run(self):
        """
        Starts the bot's main scheduling loop for live trading.
        """
        # --- Scheduling based on config ---
        print("\n--- Setting up Scheduler ---")
        # Schedule calibration for each time in the list
        for cal_time in config.CALIBRATION_TIMES:
            schedule.every().day.at(cal_time).do(self.run_calibration)
            print(f"Calibration scheduled daily at: {cal_time}")

        schedule.every(config.TRADING_SCAN_FREQUENCY_MINUTES).minutes.do(self.run_trading_scan)
        print(f"Trading scan scheduled every: {config.TRADING_SCAN_FREQUENCY_MINUTES} minutes")

        schedule.every(config.HEDGING_FREQUENCY_HOURS).hours.do(self.run_hedge_adjustment)
        print(f"Hedge adjustment scheduled every: {config.HEDGING_FREQUENCY_HOURS} hours")
        print("----------------------------")

        print("\nPerforming initial calibration before starting the schedule...")
        self.run_calibration()
        print("\nPerforming initial market scan and hedge adjustment...")
        self.run_trading_scan()
        self.run_hedge_adjustment()
        
        print("\n--- Starting Main Scheduler Loop (Live Mode) ---")
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    # This block is for running the bot in LIVE mode
    bot = TradingBot()
    bot.run()
