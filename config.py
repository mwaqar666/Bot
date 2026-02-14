import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Exchange Configuration ---
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
IS_TESTNET = True  # Set to False for real money

# --- Trading Parameters ---
SYMBOL = "BTC/USDT"  # The pair we are trading
TIMEFRAME = "5m"  # Candle timeframe
MAX_LEVERAGE = 50  # Maximum allowable leverage (safety cap)
RISK_PER_TRADE = 0.01  # Risk 1% of account balance per trade
DATA_LOOKBACK_DAYS = 1  # Number of days of data to look back

# --- Awesome Oscillator Parameters ---
AO_FAST = 5
AO_SLOW = 34

# --- Moving Average Convergence Divergence Parameters ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Relative Strength Index Parameters ---
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Stochastic RSI Parameters ---
STOCHRSI_LENGTH = 14
STOCHRSI_RSI_LENGTH = 14
STOCHRSI_K = 3
STOCHRSI_D = 3

# --- John Carter's TTM Squeeze Parameters ---
SQUEEZE_KC_LENGTH = 20
SQUEEZE_KC_SCALAR = 1.5
SQUEEZE_MOM_LENGTH = 12
SQUEEZE_MOM_SMOOTH = 6
SQUEEZE_MA_MODE = "ema"

# --- Arnaud Legoux Moving Average Parameters ---
ALMA_LENGTH = 10
ALMA_SIGMA = 6.0
ALMA_DISTRIBUTION_OFFSET = 0.85

# --- Exponential Moving Average Parameters ---
EMA_LENGTH = 20

# --- Hull Moving Average Parameters ---
HMA_LENGTH = 10

# --- Super Trend Parameters ---
SUPER_TREND_LENGTH = 7
SUPER_TREND_MULTIPLIER = 3.0

# --- Log Return Parameters ---
LOG_RETURN_LENGTH = 20

# --- Percent Return Parameters ---
PERCENT_RETURN_LENGTH = 20

# --- Entropy Parameters ---
ENTROPY_LENGTH = 10
ENTROPY_BASE = 2

# --- Mean Absolute Deviation Parameters ---
MAD_LENGTH = 30

# --- Standard Deviation Parameters ---
STD_DEV_LENGTH = 30
STD_DEV_DDOF = 1

# --- Variance Parameters ---
VARIANCE_LENGTH = 30
VARIANCE_DDOF = 0

# --- Z-Score Parameters ---
ZSCORE_LENGTH = 30
ZSCORE_STD = 1

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

# --- Donchian Channel Parameters ---
DONCHIAN_LOWER_LENGTH = 20
DONCHIAN_UPPER_LENGTH = 20

# --- Keltner Channel Parameters ---
KC_LENGTH = 20
KC_SCALAR = 2
KC_MA_MODE = "ema"

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
