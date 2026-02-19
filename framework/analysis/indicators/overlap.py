from .base import Indicator

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Super Trend
# -----------------
class SuperTrend(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)

        if super_trend is None or super_trend.empty:
            raise ValueError("SuperTrend calculation failed")

        st = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_direction = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]

        return pd.DataFrame({"st": st, "st_direction": st_direction}, index=df.index)

    def fit_scaler(self, df: pd.DataFrame) -> None:
        percentage_distance = (df["close"] - df["st"]) / df["close"]
        self.__robust_scaler.fit(percentage_distance.values.reshape(-1, 1))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["st"]) / df["close"]
        st = self.__robust_scaler.transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"st": st.flatten(), "st_direction": df["st_direction"]}, index=df.index)


# -----------------
# 2. Volume Weighted Average Price
# -----------------
class VolumeWeightedAveragePrice(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        if vwap is None or vwap.empty:
            raise ValueError("VWAP calculation failed")

        return pd.DataFrame({"vwap": vwap}, index=df.index)

    def fit_scaler(self, df: pd.DataFrame) -> None:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        self.__robust_scaler.fit(percentage_distance.values.reshape(-1, 1))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        vwap = self.__robust_scaler.transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"vwap": vwap.flatten()}, index=df.index)
