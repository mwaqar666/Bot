from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. ALMA (Arnaud Legoux Moving Average)
# -----------------
class ALMA(Indicator):
    """
    Arnaud Legoux Moving Average.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        alma = ta.alma(df["close"], length=config.ALMA_LENGTH, sigma=config.ALMA_SIGMA, distribution_offset=config.ALMA_DISTRIBUTION_OFFSET)
        if alma is not None and not alma.empty:
            df["alma"] = alma
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "alma" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        alma = df["alma"].iloc[current_idx]

        # Basic Price-MA Logic
        if price > alma:
            return SignalDirection.BUY
        elif price < alma:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 2. EMA (Exponential Moving Average)
# -----------------
class EMA(Indicator):
    """
    Exponential Moving Average.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema = ta.ema(df["close"], length=config.EMA_LENGTH)
        if ema is not None and not ema.empty:
            df["ema"] = ema
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "ema" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        ema = df["ema"].iloc[current_idx]

        if price > ema:
            return SignalDirection.BUY
        elif price < ema:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 3. HMA (Hull Moving Average)
# -----------------
class HMA(Indicator):
    """
    Hull Moving Average.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        hma = ta.hma(df["close"], length=config.HMA_LENGTH)
        if hma is not None and not hma.empty:
            df["hma"] = hma
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "hma" not in df.columns:
            return SignalDirection.NONE

        curr_hma = df["hma"].iloc[current_idx]
        prev_hma = df["hma"].iloc[current_idx - 1]

        # HMA Turning Points
        if curr_hma > prev_hma:
            return SignalDirection.BUY
        elif curr_hma < prev_hma:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 4. Super Trend
# -----------------
class SuperTrend(Indicator):
    """
    Super Trend.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)
        if super_trend is not None and not super_trend.empty:
            df["st"] = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
            df["st_direction"] = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        """
        SuperTrend Direction (1 = Uptrend, -1 = Downtrend)
        """
        if "st_direction" not in df.columns:
            return SignalDirection.NONE

        direction = df["st_direction"].iloc[current_idx]

        if direction == 1:
            return SignalDirection.BUY
        elif direction == -1:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 5. VWAP (Volume Weighted Average Price)
# -----------------
class VWAP(Indicator):
    """
    Volume Weighted Average Price.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if vwap is not None and not vwap.empty:
            df["vwap"] = vwap
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "vwap" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        vwap = df["vwap"].iloc[current_idx]

        if price > vwap:
            return SignalDirection.BUY
        elif price < vwap:
            return SignalDirection.SELL

        return SignalDirection.NONE
