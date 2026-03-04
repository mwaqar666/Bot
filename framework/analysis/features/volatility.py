from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Average True Range
# -----------------
class AverageTrueRange(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_LENGTH)

        if atr is None or atr.empty:
            raise ValueError("ATR calculation failed")

        return pd.DataFrame({"atr": atr, "atr_diff": atr.diff()}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:

        cols = ["atr", "atr_diff"]
        self._scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:

        cols = ["atr", "atr_diff"]
        norm = self._scaler.transform(df[cols])

        return pd.DataFrame(norm, columns=cols, index=df.index)


# -----------------
# 2. Normalized Average True Range
# -----------------
class NormalizedAverageTrueRange(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        natr = ta.natr(df["high"], df["low"], df["close"], length=config.ATR_LENGTH)

        if natr is None or natr.empty:
            raise ValueError("Normalized ATR calculation failed")

        return pd.DataFrame({"natr": natr}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__robust_scaler.fit(df[["natr"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        natr = self.__robust_scaler.transform(df[["natr"]])
        return pd.DataFrame({"natr": natr.flatten()}, index=df.index)


# -----------------
# 3. Bollinger Bands
# -----------------
class BollingerBands(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=config.BBANDS_LENGTH, std=config.BBANDS_STD)

        if bb is None or bb.empty:
            raise ValueError("Bollinger Bands calculation failed")

        bb_lower = bb[f"BBL_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_mid = bb[f"BBM_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_upper = bb[f"BBU_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_width = bb[f"BBB_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_pct = bb[f"BBP_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]

        bb_lower_diff = bb_lower.diff()
        bb_mid_diff = bb_mid.diff()
        bb_upper_diff = bb_upper.diff()
        bb_width_diff = bb_width.diff()
        bb_pct_diff = bb_pct.diff()

        return pd.DataFrame(
            {
                "bb_lower": bb_lower,
                "bb_mid": bb_mid,
                "bb_upper": bb_upper,
                "bb_width": bb_width,
                "bb_pct": bb_pct,
                "bb_lower_diff": bb_lower_diff,
                "bb_mid_diff": bb_mid_diff,
                "bb_upper_diff": bb_upper_diff,
                "bb_width_diff": bb_width_diff,
                "bb_pct_diff": bb_pct_diff,
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:

        cols = ["bb_width", "bb_pct", "bb_width_diff", "bb_pct_diff"]
        self._scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:

        cols = ["bb_width", "bb_pct", "bb_width_diff", "bb_pct_diff"]
        norm = self._scaler.transform(df[cols])

        return pd.DataFrame(norm, columns=cols, index=df.index)


# -----------------
# 4. Ulcer Index
# -----------------
class UlcerIndex(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = ta.ui(df["close"], length=config.UI_LENGTH, scalar=config.UI_SCALAR)

        if ui is None or ui.empty:
            raise ValueError("Ulcer Index calculation failed")

        return pd.DataFrame({"ui": ui}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__robust_scaler.fit(df[["ui"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = self.__robust_scaler.transform(df[["ui"]])
        return pd.DataFrame({"ui": ui.flatten()}, index=df.index)
