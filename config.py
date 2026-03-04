import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Configuration ---
MODE = "data"  # Default mode to run the bot in: 'trade' for live bot, 'data' for downloading/analyzing data.

# --- Exchange Configuration ---
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
IS_TESTNET = True  # Set to False for real money

# --- Trading Parameters ---
SYMBOL = "BTC/USDT"  # The pair we are trading
TIMEFRAME = "5m"  # Candle timeframe
MAX_LEVERAGE = 50  # Maximum allowable leverage (safety cap)
RISK_PER_TRADE = 0.01  # Risk 1% of account balance per trade
DATA_LOOKBACK_DAYS = 5  # Number of days of data to look back

# --- Moving Average Convergence Divergence Parameters ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Relative Strength Index Parameters ---
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- John Carter's TTM Squeeze Parameters ---
SQUEEZE_KC_LENGTH = 20
SQUEEZE_KC_SCALAR = 1.5
SQUEEZE_MOM_LENGTH = 12
SQUEEZE_MOM_SMOOTH = 6
SQUEEZE_MA_MODE = "ema"

# --- Super Trend Parameters ---
SUPER_TREND_LENGTH = 7
SUPER_TREND_MULTIPLIER = 3.0

# --- EMA Parameters ---
EMA_FAST_LENGTH = 14
EMA_SLOW_LENGTH = 28

# --- Log Return Parameters ---
LOG_RETURN_LENGTH = 20

# --- Entropy Parameters ---
ENTROPY_LENGTH = 10
ENTROPY_BASE = 2

# --- ADX Parameters ---
ADX_LENGTH = 14

# --- Parabolic Stop and Reverse Parameters ---
PSAR_INIT_ACC = 0.02
PSAR_ACC = 0.02
PSAR_MAX_ACC = 0.2

# --- Average True Range Parameters ---
ATR_LENGTH = 14

# --- Bollinger Bands Parameters ---
BBANDS_LENGTH = 20
BBANDS_STD = 2.0

# --- Ulcer Index Parameters ---
UI_LENGTH = 14
UI_SCALAR = 100

# --- Chaikin Money Flow Parameters ---
CMF_LENGTH = 20

# --- Money Flow Index Parameters ---
MFI_LENGTH = 14

# --- Exit Multipliers ---
SL_ATR_MULTIPLIER = 1.0  # Stop Loss distance = 1 * ATR
TP_ATR_MULTIPLIER = 2.0  # Take Profit distance = 2 * ATR
