import pandas as pd
import json
import os
import datetime

class Portfolio:
    """
    Manages a fictional trading portfolio, tracking cash, positions, and performance.
    Handles both long and short positions and persists its state to a file.
    """
    def __init__(self, initial_cash=100000.0, state_file="portfolio_state.json"):
        """
        Initializes the portfolio, loading from state_file if it exists.
        """
        self.state_file = state_file
        # Default initial state
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.stock_holdings = {}
        self.trade_log = []
        
        # Attempt to load a previous state
        if not self.load_state():
            print(f"No state file found. Initializing new portfolio with ${self.initial_cash:,.2f} cash.")

    def save_state(self):
        """Saves the current portfolio state to the state_file."""
        try:
            state = {
                "initial_cash": self.initial_cash,
                "cash": self.cash,
                "positions": self.positions,
                "stock_holdings": self.stock_holdings,
                "trade_log": self.trade_log
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"!!! CRITICAL: Failed to save portfolio state: {e} !!!")
            
    def load_state(self):
        """Loads portfolio state from the state_file if it exists."""
        if not os.path.exists(self.state_file):
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.initial_cash = state.get("initial_cash", self.initial_cash)
            self.cash = state.get("cash", self.cash)
            self.positions = state.get("positions", {})
            self.stock_holdings = state.get("stock_holdings", {})
            self.trade_log = state.get("trade_log", [])
            
            print(f"--- Portfolio state successfully loaded from {self.state_file} ---")
            self.get_pnl() # Display loaded summary
            return True
        except Exception as e:
            print(f"!!! CRITICAL: Failed to load portfolio state from {self.state_file}: {e} !!!")
            return False

    def _log_trade(self, timestamp, trade_type, symbol, quantity, price, cost):
        """
        Logs a transaction to the trade history.
        """
        # Convert timestamp to string for JSON serialization
        ts_str = timestamp.isoformat() if isinstance(timestamp, datetime.datetime) else str(timestamp)
        
        self.trade_log.append({
            "timestamp": ts_str, "trade_type": trade_type, "symbol": symbol,
            "quantity": quantity, "price": price, "cost": cost,
            "cash_balance": self.cash
        })

    def transact_option(self, contract_id, quantity_change, price, timestamp):
        """
        Handles buying or selling options, allowing for short positions.
        """
        cost = quantity_change * price * 100

        if self.cash < cost:
            print(f"Error: Not enough cash for transaction. Required: ${cost:,.2f}, Available: ${self.cash:,.2f}")
            return

        self.cash -= cost
        trade_type = "BUY_OPTION" if quantity_change > 0 else "SELL_OPTION"
        
        existing_position = self.positions.get(contract_id)
        
        if not existing_position:
            self.positions[contract_id] = {
                "quantity": quantity_change,
                "entry_price": price,
                "entry_time": timestamp.isoformat()
            }
            action = "Bought" if quantity_change > 0 else "Shorted"
            print(f"New Position: {action} {abs(quantity_change)} of {contract_id} at ${price:.2f}")
        else:
            new_quantity = existing_position['quantity'] + quantity_change
            if new_quantity == 0:
                pnl = (price - existing_position['entry_price']) * existing_position['quantity'] * 100
                print(f"Closing Position: Transacted {quantity_change} of {contract_id} at ${price:.2f}. Realized P&L: ${pnl:.2f}")
                del self.positions[contract_id]
            else:
                new_avg_price = (existing_position['quantity'] * existing_position['entry_price'] + quantity_change * price) / new_quantity
                existing_position['quantity'] = new_quantity
                existing_position['entry_price'] = new_avg_price
                action = "Added to" if quantity_change > 0 else "Reduced"
                print(f"Position Update: {action} {contract_id}. New quantity: {new_quantity}")

        self._log_trade(timestamp, trade_type, contract_id, abs(quantity_change), price, cost)
        self.save_state() # Persist state after every transaction

    def update_stock_position(self, stock_ticker, quantity_change, price, timestamp):
        """
        Adjusts the number of shares held for delta hedging.
        """
        cost = quantity_change * price
        
        if self.cash < cost:
            print(f"Error: Not enough cash for stock hedge. Required: ${cost:,.2f}, Available: ${self.cash:,.2f}")
            return
            
        self.cash -= cost
        self.stock_holdings[stock_ticker] = self.stock_holdings.get(stock_ticker, 0) + quantity_change
        
        trade_type = "BUY_STOCK" if quantity_change > 0 else "SELL_STOCK"
        self._log_trade(timestamp, trade_type, stock_ticker, abs(quantity_change), price, cost)
        action = "Bought" if quantity_change > 0 else "Sold"
        print(f"Hedge: {action} {abs(quantity_change)} shares of {stock_ticker} at ${price:.2f}. New holding: {self.stock_holdings[stock_ticker]}")
        self.save_state() # Persist state after every transaction

    def get_pnl(self, S0=None, options_market_data=None):
        """
        Calculates and displays the current portfolio summary, including unrealized P&L.

        Args:
            S0 (float, optional): Current price of the underlying stock.
            options_market_data (pd.DataFrame, optional):
                A DataFrame with current market data for options,
                including 'contract_id' and 'price' columns.
        """
        realized_pnl = self.cash - self.initial_cash
        unrealized_options_pnl = 0.0
        options_market_value = 0.0

        print("\n--- Portfolio Summary ---")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Current Cash: ${self.cash:,.2f}")

        print("\nOpen Option Positions:")
        if not self.positions:
            print("  None")
        else:
            # Use a copy to avoid modifying the original dict via references
            pos_df = pd.DataFrame.from_dict(self.positions, orient='index')
            pos_df.index.name = 'contract_id'

            if options_market_data is not None and not options_market_data.empty:
                market_prices = options_market_data.set_index('contract_id')['price']
                pos_df['market_price'] = market_prices
                pos_df['market_price'] = pos_df['market_price'].fillna(-1)  # Mark positions not found

                # Calculate unrealized P&L where market price is available
                valid_prices = pos_df['market_price'] != -1
                pos_df['market_value'] = pos_df['quantity'] * pos_df['market_price'] * 100
                pos_df['cost_basis'] = pos_df['quantity'] * pos_df['entry_price'] * 100
                pos_df.loc[valid_prices, 'unrealized_pnl'] = pos_df['market_value'] - pos_df['cost_basis']
                unrealized_options_pnl = pos_df.loc[valid_prices, 'unrealized_pnl'].sum()
                options_market_value = pos_df.loc[valid_prices, 'market_value'].sum()

            for contract_id, details in self.positions.items():
                pos_type = "LONG" if details['quantity'] > 0 else "SHORT"
                pnl_str = ""
                if 'unrealized_pnl' in pos_df.columns and contract_id in pos_df.index:
                    pnl_val = pos_df.loc[contract_id, 'unrealized_pnl']
                    if pd.notna(pnl_val):
                        pnl_str = f" | Unrealized P&L: ${pnl_val:,.2f}"
                    else:
                        pnl_str = " | Unrealized P&L: (Market Price N/A)"
                print(f"  - {pos_type} {abs(details['quantity'])} {contract_id} @ avg price ${details['entry_price']:.2f}{pnl_str}")

        stock_market_value = 0
        print("\nStock Holdings (for Hedging):")
        if not self.stock_holdings or all(q == 0 for q in self.stock_holdings.values()):
            print("  None")
        else:
            for stock, quantity in self.stock_holdings.items():
                if quantity != 0:
                    pos_type = "LONG" if quantity > 0 else "SHORT"
                    market_value_str = ""
                    if S0 is not None:
                        value = quantity * S0
                        stock_market_value += value
                        market_value_str = f" | Market Value: ${value:,.2f}"
                    print(f"  - {pos_type} {abs(quantity)} shares of {stock}{market_value_str}")

        print("\n--- P&L Overview ---")
        print(f"Realized P&L (from cash change): ${realized_pnl:,.2f}")

        if unrealized_options_pnl != 0.0 or stock_market_value != 0.0:
            print(f"Unrealized P&L (open options): ${unrealized_options_pnl:,.2f}")

            # Total portfolio value = cash + value of all open positions
            total_equity = self.cash + stock_market_value + options_market_value
            portfolio_return = ((total_equity - self.initial_cash) / self.initial_cash) * 100

            print(f"Total Portfolio Value (Equity): ${total_equity:,.2f}")
            print(f"Total Return vs Initial Cash: {portfolio_return:.2f}%")

        print("-------------------------\n")

        return realized_pnl
