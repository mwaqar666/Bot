from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. Entropy
# -----------------
class Entropy(Indicator):
    """
    Entropy.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = ta.entropy(df["close"], length=config.ENTROPY_LENGTH, base=config.ENTROPY_BASE)
        if entropy is not None and not entropy.empty:
            df["entropy"] = entropy
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Entropy describes disorder, not direction.
        # Typically used to filter (High Entropy = Don't Trade).
        return SignalDirection.NONE


# -----------------
# 2. Mean Absolute Deviation (MAD)
# -----------------
class MAD(Indicator):
    """
    Mean Absolute Deviation.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mad = ta.mad(df["close"], length=config.MAD_LENGTH)
        if mad is not None and not mad.empty:
            df["mad"] = mad
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Measure of volatility/dispersion.
        return SignalDirection.NONE


# -----------------
# 3. Standard Deviation (StdDev)
# -----------------
class StandardDeviation(Indicator):
    """
    Standard Deviation.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        stdev = ta.stdev(df["close"], length=config.STD_DEV_LENGTH, ddof=config.STD_DEV_DDOF)
        if stdev is not None and not stdev.empty:
            df["stdev"] = stdev
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Measure of volatility.
        return SignalDirection.NONE


# -----------------
# 4. Z-Score
# -----------------
class ZScore(Indicator):
    """
    Z-Score.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        zscore = ta.zscore(df["close"], length=config.ZSCORE_LENGTH, std=config.ZSCORE_STD)
        if zscore is not None and not zscore.empty:
            df["zscore"] = zscore
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "zscore" not in df.columns:
            return SignalDirection.NONE

        val = df["zscore"].iloc[current_idx]

        # Mean Reversion Logic
        if val < -2:
            return SignalDirection.BUY  # Oversold
        elif val > 2:
            return SignalDirection.SELL  # Overbought

        return SignalDirection.NONE
