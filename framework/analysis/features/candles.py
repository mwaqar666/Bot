from .base import Feature
from typing_extensions import Self

import pandas as pd


# -----------------
# 1. Candle Structure
# -----------------
class CandleStructure(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_body = (df["close"] - df["open"]) / df["open"]
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
        candle_range = (df["high"] - df["low"]) / df["low"]

        return pd.DataFrame({"rel_body": rel_body, "upper_wick": upper_wick, "lower_wick": lower_wick, "candle_range": candle_range}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self._check_scaler()

        cols = ["rel_body", "upper_wick", "lower_wick", "candle_range"]
        self._scaler.fit(df[cols])

        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_scaler()

        cols = ["rel_body", "upper_wick", "lower_wick", "candle_range"]
        norm = self._scaler.transform(df[cols])

        return pd.DataFrame(norm, columns=cols, index=df.index)
