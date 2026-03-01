from .base import Indicator
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Draw Down
# -----------------
class DrawDown(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        draw_down = ta.drawdown(df["close"])

        if draw_down is None or draw_down.empty:
            raise ValueError("Drawdown calculation failed")

        dd = draw_down["DD"]
        dd_pct = draw_down["DD_PCT"] * -1
        dd_log = draw_down["DD_LOG"] * -1

        return pd.DataFrame({"dd": dd, "dd_pct": dd_pct, "dd_log": dd_log}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__robust_scaler.fit(df[["dd_pct"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        dd_pct = self.__robust_scaler.transform(df[["dd_pct"]]).clip(-5, 5)
        return pd.DataFrame({"dd_pct": dd_pct.flatten()}, index=df.index)


# -----------------
# 2. Log Return
# -----------------
class LogReturn(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = ta.log_return(df["close"], length=config.LOG_RETURN_LENGTH)

        if log_return is None or log_return.empty:
            raise ValueError("Log return calculation failed")

        return pd.DataFrame({"log_return": log_return}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__robust_scaler.fit(df[["log_return"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = self.__robust_scaler.transform(df[["log_return"]]).clip(-5, 5)
        return pd.DataFrame({"log_return": log_return.flatten()}, index=df.index)
