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

# --- Even Better Sine Wave Parameters ---
EBSW_LENGTH = 40
EBSW_BARS = 10

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

# --- Percentage Volume Oscillator Parameters ---
PVO_FAST = 12
PVO_SLOW = 26
PVO_SIGNAL = 9

# --- Williams %R Parameters ---
WILLR_LENGTH = 14

# --- Super Trend Parameters ---
SUPER_TREND_LENGTH = 7
SUPER_TREND_MULTIPLIER = 3.0

# --- Log Return Parameters ---
LOG_RETURN_LENGTH = 20

# --- Entropy Parameters ---
ENTROPY_LENGTH = 10
ENTROPY_BASE = 2

# --- ADX Parameters ---
ADX_LENGTH = 14

# --- Aroon Parameters ---
AROON_LENGTH = 14
AROON_SCALAR = 100

# --- Choppiness Index Parameters ---
CHOP_LENGTH = 14
CHOP_ATR_LENGTH = 1
CHOP_LN = False
CHOP_SCALAR = 100

# --- Parabolic Stop and Reverse Parameters ---
PSAR_INIT_ACC = 0.02
PSAR_ACC = 0.02
PSAR_MAX_ACC = 0.2

# --- Vortex Parameters ---
VORTEX_LENGTH = 14

# --- Average True Range Parameters ---
ATR = 14

# --- Bollinger Bands Parameters ---
BBANDS = 20
BBANDS_STD = 2.0

# --- Ulcer Index Parameters ---
UI_LENGTH = 14
UI_SCALAR = 100

# --- CMF Parameters ---
CMF_LENGTH = 20

# --- EFI Parameters ---
EFI_LENGTH = 13

# --- MFI Parameters ---
MFI_LENGTH = 14

# --- VP Parameters ---
VP_LENGTH = 10

# --- Exit Multipliers ---
SL_ATR_MULTIPLIER = 1.0  # Stop Loss distance = 1 * ATR
TP_ATR_MULTIPLIER = 2.0  # Take Profit distance = 2 * ATR
