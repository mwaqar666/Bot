from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. ATR (Average True Range)
# -----------------
class ATR(Indicator):
    """
    Average True Range.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR)
        if atr is not None and not atr.empty:
            df["atr"] = atr
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Volatility Measure (for Stop Loss/Position Sizing).
        return SignalDirection.NONE


# -----------------
# 2. Bollinger Bands
# -----------------
class BollingerBands(Indicator):
    """
    Bollinger Bands.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=config.BBANDS, std=config.BBANDS_STD)
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb[f"BBL_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_mid"] = bb[f"BBM_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_upper"] = bb[f"BBU_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_width"] = bb[f"BBB_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_pct"] = bb[f"BBP_{config.BBANDS}_{config.BBANDS_STD}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "bb_lower" not in df.columns or "bb_upper" not in df.columns:
            return SignalDirection.NONE

        close = df["close"].iloc[current_idx]
        bb_lower = df["bb_lower"].iloc[current_idx]
        bb_upper = df["bb_upper"].iloc[current_idx]

        # Mean Reversion Logic
        if close < bb_lower:
            return SignalDirection.BUY  # Oversold
        elif close > bb_upper:
            return SignalDirection.SELL  # Overbought

        return SignalDirection.NONE


# -----------------
# 3. Ulcer Index
# -----------------
class UlcerIndex(Indicator):
    """
    Ulcer Index.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = ta.ui(df["close"], length=config.UI_LENGTH, scalar=config.UI_SCALAR)
        if ui is not None and not ui.empty:
            df["ui"] = ui
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Risk Metric. High UI = High Drawdown Risk.
        return SignalDirection.NONE
