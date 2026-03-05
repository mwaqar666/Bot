from .base import Feature
from typing_extensions import Self

import pandas as pd
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Candle Structure
# -----------------
class CandleStructure(Feature):
    __cols = ["rel_body", "upper_wick", "lower_wick"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_body = (df["close"] - df["open"]) / df["open"]
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

        return pd.DataFrame(
            {
                "rel_body": rel_body,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        candle_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(candle_norm, columns=self.__cols, index=df.index)
