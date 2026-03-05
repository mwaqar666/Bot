from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import QuantileTransformer, RobustScaler


# -----------------
# 1. Super Trend
# -----------------
class SuperTrend(Feature):
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)

        if super_trend is None or super_trend.empty:
            raise ValueError("SuperTrend calculation failed")

        st = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_direction = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_dist = (df["close"] - st) / df["close"]

        # Drop rows where st is 0 or NaN before returning
        mask = (st != 0) & st.notna()

        return pd.DataFrame(
            {
                "st": st[mask],
                "st_direction": st_direction[mask],
                "st_dist": st_dist[mask],
            },
            index=df.index[mask],
        )

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["st_dist"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        st_dist = self.__scaler.transform(df[["st_dist"]])

        return pd.DataFrame(
            {
                "st_dist": st_dist.flatten(),
                "st_direction": df["st_direction"].astype(float),
            },
            index=df.index,
        )


# -----------------
# 2. Volume Weighted Average Price
# -----------------
class VolumeWeightedAveragePrice(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        if vwap is None or vwap.empty:
            raise ValueError("VWAP calculation failed")

        return pd.DataFrame({"vwap": vwap}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        self.__robust_scaler.fit(percentage_distance.values.reshape(-1, 1))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        vwap = self.__robust_scaler.transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"vwap": vwap.flatten()}, index=df.index)


# -----------------
# 3. Exponential Moving Average
# -----------------
class ExponentialMovingAverage(Feature):
    __cols = ["ema_spread", "ema_spread_diff"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = ta.ema(df["close"], length=config.EMA_FAST_LENGTH)
        ema_slow = ta.ema(df["close"], length=config.EMA_SLOW_LENGTH)

        if ema_fast is None or ema_fast.empty or ema_slow is None or ema_slow.empty:
            raise ValueError("EMA calculation failed")

        ema_spread = (ema_fast - ema_slow) / ema_slow

        return pd.DataFrame(
            {
                "ema_spread": ema_spread,
                "ema_spread_diff": ema_spread.diff(),
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(scaled, columns=self.__cols, index=df.index)
