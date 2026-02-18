from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Draw Down
# -----------------
class DrawDown(Indicator):
    """
    Draw Down.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        draw_down = ta.drawdown(df["close"])

        if draw_down is None or draw_down.empty:
            raise ValueError("Drawdown calculation failed")

        dd = draw_down["DD"]
        dd_pct = draw_down["DD_PCT"] * -1
        dd_log = draw_down["DD_LOG"] * -1

        return pd.DataFrame({"dd": dd, "dd_pct": dd_pct, "dd_log": dd_log}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Drawdown is a risk metric, generally not a signal generator.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        dd_pct = self.__robust_scaler.fit_transform(df[["dd_pct"]])
        return pd.DataFrame({"dd_pct": dd_pct.flatten()}, index=df.index)


# -----------------
# 2. Log Return
# -----------------
class LogReturn(Indicator):
    """
    Log Return.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = ta.log_return(df["close"], length=config.LOG_RETURN_LENGTH)

        if log_return is None or log_return.empty:
            raise ValueError("Log return calculation failed")

        return pd.DataFrame({"log_return": log_return}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Returns are features for ML, not typically direct signals.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = self.__robust_scaler.fit_transform(df[["log_return"]])
        return pd.DataFrame({"log_return": log_return.flatten()}, index=df.index)
