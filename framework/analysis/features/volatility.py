from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Average True Range
# -----------------
class AverageTrueRange(Feature):
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_LENGTH)

        if atr is None or atr.empty:
            raise ValueError("Average True Range calculation failed")

        natr = atr * (100 / df["close"])

        return pd.DataFrame({"atr": atr, "natr": natr}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["natr"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        natr_norm = self.__scaler.transform(df[["natr"]])
        return pd.DataFrame({"atr": df["atr"], "natr": natr_norm.flatten()}, index=df.index)


# -----------------
# 2. Bollinger Bands
# -----------------
class BollingerBands(Feature):
    __cols = ["bb_width"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=config.BBANDS_LENGTH, std=config.BBANDS_STD)

        if bb is None or bb.empty:
            raise ValueError("Bollinger Bands calculation failed")

        bb_lower = bb[f"BBL_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_mid = bb[f"BBM_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_upper = bb[f"BBU_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_width = bb[f"BBB_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]
        bb_pct = bb[f"BBP_{config.BBANDS_LENGTH}_{config.BBANDS_STD}"]

        return pd.DataFrame({"bb_lower": bb_lower, "bb_mid": bb_mid, "bb_upper": bb_upper, "bb_width": bb_width, "bb_pct": bb_pct}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        bb_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(bb_norm, columns=self.__cols, index=df.index)


# -----------------
# 3. Ulcer Index
# -----------------
class UlcerIndex(Feature):
    __cols = ["ui"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ui = ta.ui(df["close"], length=config.UI_LENGTH, scalar=config.UI_SCALAR)

        if ui is None or ui.empty:
            raise ValueError("Ulcer Index calculation failed")

        return pd.DataFrame({"ui": ui}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ui_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(ui_norm, columns=self.__cols, index=df.index)
