from .base import Indicator

import config

import pandas as pd
from sklearn.preprocessing import RobustScaler

# Relative Body: (Close - Open) / Open (The Green/Red strength).
# Upper Wick: (High - max(Open, Close)) / Open (The rejection from the top).
# Lower Wick: (min(Open, Close) - Low) / Open (The support from the bottom).


# -----------------
# 1. Intra Day Candle
# -----------------
class IntraDayCandle(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_body = (df["close"] - df["open"]) / df["open"]
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

        return pd.DataFrame({"rel_body": rel_body, "upper_wick": upper_wick, "lower_wick": lower_wick}, index=df.index)

    def fit_scaler(self, df: pd.DataFrame) -> None:
        cols = ["rel_body", "upper_wick", "lower_wick"]
        self.__robust_scaler.fit(df[cols])

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["rel_body", "upper_wick", "lower_wick"]
        candles = self.__robust_scaler.transform(df[cols])

        return pd.DataFrame(candles, columns=cols, index=df.index)
