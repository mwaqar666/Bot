from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. Draw Down
# -----------------
class DrawDown(Indicator):
    """
    Draw Down.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        draw_down = ta.drawdown(df["close"])
        if draw_down is not None and not draw_down.empty:
            df["draw_down"] = draw_down["DD"]
            df["draw_down_pct"] = draw_down["DD_PCT"]
            df["draw_down_log"] = draw_down["DD_LOG"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Drawdown is a risk metric, generally not a signal generator.
        return SignalDirection.NONE


# -----------------
# 2. Log Return
# -----------------
class LogReturn(Indicator):
    """
    Log Return.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = ta.log_return(df["close"], length=config.LOG_RETURN_LENGTH)
        if log_return is not None and not log_return.empty:
            df["log_return"] = log_return
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Returns are features for ML, not typically direct signals.
        return SignalDirection.NONE
