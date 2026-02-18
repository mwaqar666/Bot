from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Entropy
# -----------------
class Entropy(Indicator):
    """
    Entropy.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = ta.entropy(df["close"], length=config.ENTROPY_LENGTH, base=config.ENTROPY_BASE)

        if entropy is None or entropy.empty:
            raise ValueError("Entropy calculation failed")

        return pd.DataFrame({"entropy": entropy}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Entropy describes disorder, not direction.
        # Typically used to filter (High Entropy = Don't Trade).
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = self.__min_max_scaler.fit_transform(df[["entropy"]])
        return pd.DataFrame({"entropy": entropy.flatten()}, index=df.index)


# -----------------
# 2. Mean Absolute Deviation
# -----------------
class MeanAbsoluteDeviation(Indicator):
    """
    Mean Absolute Deviation.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mad = ta.mad(df["close"], length=config.MAD_LENGTH)

        if mad is None or mad.empty:
            raise ValueError("Mean absolute deviation calculation failed")

        return pd.DataFrame({"mad": mad}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Measure of volatility/dispersion.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({}, index=df.index)


# -----------------
# 3. Standard Deviation (StdDev)
# -----------------
class StandardDeviation(Indicator):
    """
    Standard Deviation.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        stdev = ta.stdev(df["close"], length=config.STD_DEV_LENGTH, ddof=config.STD_DEV_DDOF)

        if stdev is None or stdev.empty:
            raise ValueError("Standard deviation calculation failed")

        return pd.DataFrame({"stdev": stdev}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Measure of volatility.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({}, index=df.index)


# -----------------
# 4. Z-Score
# -----------------
class ZScore(Indicator):
    """
    Z-Score.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        zscore = ta.zscore(df["close"], length=config.ZSCORE_LENGTH, std=config.ZSCORE_STD)

        if zscore is None or zscore.empty:
            raise ValueError("Z-score calculation failed")

        return pd.DataFrame({"zscore": zscore}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "zscore" not in df.columns:
            return SignalDirection.NONE

        val = df["zscore"].iloc[current_idx]

        # Mean Reversion Logic
        if val < -2:
            return SignalDirection.BUY  # Oversold
        elif val > 2:
            return SignalDirection.SELL  # Overbought

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        zscore = self.__robust_scaler.fit_transform(df[["zscore"]])
        return pd.DataFrame({"zscore": zscore.flatten()}, index=df.index)
