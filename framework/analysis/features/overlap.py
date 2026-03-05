from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Super Trend
# -----------------
class SuperTrend(Feature):
    __cols = ["st_dist"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)

        if st is None or st.empty:
            raise ValueError("SuperTrend calculation failed")

        st_line = st[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_dist = (df["close"] - st_line) / df["close"]

        mask = (st_line != 0) & st_line.notna()

        return pd.DataFrame({"st_line": st_line[mask], "st_dist": st_dist[mask]}, index=df.index[mask])

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        st_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(st_norm, columns=self.__cols, index=df.index)


# -----------------
# 2. Exponential Moving Average
# -----------------
class ExponentialMovingAverage(Feature):
    __cols = ["ema_spread"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = ta.ema(df["close"], length=config.EMA_FAST_LENGTH)
        ema_slow = ta.ema(df["close"], length=config.EMA_SLOW_LENGTH)

        if ema_fast is None or ema_fast.empty or ema_slow is None or ema_slow.empty:
            raise ValueError("EMA calculation failed")

        ema_spread = (ema_fast - ema_slow) / ema_slow

        return pd.DataFrame({"ema_spread": ema_spread}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(ema_norm, columns=self.__cols, index=df.index)
