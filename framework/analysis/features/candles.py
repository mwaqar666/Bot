from .base import Feature
from typing_extensions import Self

import pandas as pd
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Candle Structure
# -----------------
class CandleStructure(Feature):
    __cols = ["rel_body", "upper_wick", "lower_wick", "candle_range", "rel_body_diff"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_body = (df["close"] - df["open"]) / df["open"]
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
        candle_range = (df["high"] - df["low"]) / df["low"]

        return pd.DataFrame(
            {
                "rel_body": rel_body,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "candle_range": candle_range,
                "rel_body_diff": rel_body.diff(),
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(norm, columns=self.__cols, index=df.index)
