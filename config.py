import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Exchange Configuration ---
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
IS_TESTNET = True  # Set to False for real money

# --- Trading Parameters ---
SYMBOL = "BTC/USDT:USDT"  # The pair we are trading
TIMEFRAME = "15m"  # 15 minute candles
MAX_LEVERAGE = 50  # Maximum allowable leverage (safety cap)
RISK_PER_TRADE = 0.01  # Risk 1% of account balance per trade

# --- Strategy Parameters ---
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2.0

# Thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_THRESHOLD = 25  # Minimum trend strength required

# Exit Multipliers
SL_ATR_MULTIPLIER = 2.0  # Stop Loss distance = 2 * ATR
TP_ATR_MULTIPLIER = 4.0  # Take Profit distance = 4 * ATR
