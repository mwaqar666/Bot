from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Draw Down
# -----------------
class DrawDown(Feature):
    __cols = ["dd_pct"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        dd = ta.drawdown(df["close"])

        if dd is None or dd.empty:
            raise ValueError("Drawdown calculation failed")

        dd_line = dd["DD"]
        dd_pct = dd["DD_PCT"]

        return pd.DataFrame({"dd_line": dd_line, "dd_pct": dd_pct}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dd_pct = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(dd_pct, columns=self.__cols, index=df.index)


# -----------------
# 2. Log Return
# -----------------
class LogReturn(Feature):
    __cols = ["log_return"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = ta.log_return(df["close"], length=config.LOG_RETURN_LENGTH)

        if log_return is None or log_return.empty:
            raise ValueError("Log return calculation failed")

        return pd.DataFrame({"log_return": log_return}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(log_return_norm, columns=self.__cols, index=df.index)
