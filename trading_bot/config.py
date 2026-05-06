# --- Main Configuration ---
TICKER_SYMBOL = "SPY"
INITIAL_CASH = 100000.0

# --- Scheduling Configuration (in minutes) ---
# Note: schedule library uses minutes, hours, days syntax.
# We'll define the schedule directly in trader.py but keep frequencies here.
# CALIBRATION_FREQUENCY_HOURS = 4  # Switched to a specific time
CALIBRATION_TIMES = ["10:05", "12:00", "14:00"]  # Times to run calibration every day
TRADING_SCAN_FREQUENCY_MINUTES = 30
HEDGING_FREQUENCY_HOURS = 1

# --- Calibration Data Configuration ---
CALIBRATION_TARGETS_DAYS = [7, 14, 45, 365]
CALIBRATION_STRIKE_WINDOW_PCT = 0.08
CALIBRATION_MIN_OPTIONS_PER_EXPIRY = 8
CALIBRATION_MIN_VOLUME = 20

# --- Trading Scan Configuration ---
SCAN_CONFIG = {
    'strike_window_pct': 0.08,
    'min_volume': 50,
    'min_open_interest': 100,
    'min_days_to_expiry': 18,
    'max_days_to_expiry': 180,
}

# --- Trading Logic Configuration ---
# An opportunity is significant if the theoretical profit is greater than X times the model's calibration RMSE
PROFIT_THRESHOLD_MULTIPLIER = 3.0
# Close a position if the market price is within this percentage of the model price
CLOSE_POSITION_THRESHOLD_PCT = 0.05 

# --- Optimizer Configuration ---
OPTIMIZER_BOUNDS = [
    (0.001, 0.9), (1.0, 8.0), (0.01, 2.0), (-0.9, -0.3), (0.1, 1.5),  # Heston Params
    (0.01, 2.0), (-0.5, 0.1), (0.0001, 0.5)                         # Jump Params
]
OPTIMIZER_SETTINGS = {
    'maxiter': 2000, 'popsize': 15, 'tol': 0.01, 'updating': 'deferred', 'workers': -1
}
