from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Average True Range
# -----------------
class AverageTrueRange(Indicator):
    """
    Average True Range.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR)

        if atr is None or atr.empty:
            raise ValueError("ATR calculation failed")

        return pd.DataFrame({"atr": atr}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Volatility Measure (for Stop Loss/Position Sizing).
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        norm_atr = df["atr"] / df["close"]
        atr = self.__robust_scaler.fit_transform(norm_atr.values.reshape(-1, 1))

        return pd.DataFrame({"atr": atr.flatten()}, index=df.index)


# -----------------
# 2. Bollinger Bands
# -----------------
class BollingerBands(Indicator):
    """
    Bollinger Bands.
    """

    __robust_scaler = RobustScaler()
    __min_max_scaler = MinMaxScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=config.BBANDS, std=config.BBANDS_STD)

        if bb is None or bb.empty:
            raise ValueError("Bollinger Bands calculation failed")

        bb_lower = bb[f"BBL_{config.BBANDS}_{config.BBANDS_STD}"]
        bb_mid = bb[f"BBM_{config.BBANDS}_{config.BBANDS_STD}"]
        bb_upper = bb[f"BBU_{config.BBANDS}_{config.BBANDS_STD}"]
        bb_width = bb[f"BBB_{config.BBANDS}_{config.BBANDS_STD}"]
        bb_pct = bb[f"BBP_{config.BBANDS}_{config.BBANDS_STD}"]

        return pd.DataFrame({"bb_lower": bb_lower, "bb_mid": bb_mid, "bb_upper": bb_upper, "bb_width": bb_width, "bb_pct": bb_pct}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "bb_lower" not in df.columns or "bb_upper" not in df.columns:
            return SignalDirection.NONE

        close = df["close"].iloc[current_idx]
        bb_lower = df["bb_lower"].iloc[current_idx]
        bb_upper = df["bb_upper"].iloc[current_idx]

        # Mean Reversion Logic
        if close < bb_lower:
            return SignalDirection.BUY  # Oversold
        elif close > bb_upper:
            return SignalDirection.SELL  # Overbought

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        bb_width = self.__robust_scaler.fit_transform(df[["bb_width"]])
        bb_pct = self.__min_max_scaler.fit_transform(df[["bb_pct"]])

        return pd.DataFrame({"bb_width": bb_width.flatten(), "bb_pct": bb_pct.flatten()}, index=df.index)


# -----------------
# 3. Ulcer Index
# -----------------
class UlcerIndex(Indicator):
    """
    Ulcer Index.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = ta.ui(df["close"], length=config.UI_LENGTH, scalar=config.UI_SCALAR)

        if ui is None or ui.empty:
            raise ValueError("Ulcer Index calculation failed")

        return pd.DataFrame({"ui": ui}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Risk Metric. High UI = High Drawdown Risk.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = self.__robust_scaler.fit_transform(df[["ui"]])
        return pd.DataFrame({"ui": ui.flatten()}, index=df.index)
