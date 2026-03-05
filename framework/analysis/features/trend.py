from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from sklearn.preprocessing import QuantileTransformer


# -----------------
# 1. Average Directional Index
# -----------------
class AverageDirectionalIndex(Feature):
    __cols = ["adx"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_data = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_LENGTH)

        if adx_data is None or adx_data.empty:
            raise ValueError("ADX calculation failed")

        dmp = adx_data[f"DMP_{config.ADX_LENGTH}"]
        dmn = adx_data[f"DMN_{config.ADX_LENGTH}"]

        adx = adx_data[f"ADX_{config.ADX_LENGTH}"]
        adx_osc = dmp - dmn

        return pd.DataFrame({"adx": adx, "adx_osc": adx_osc}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(adx_norm, columns=self.__cols, index=df.index)


# -----------------
# 2. Parabolic Stop and Reverse
# -----------------
class ParabolicStopAndReverse(Feature):
    __cols = ["psar_dist"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        psar = ta.psar(df["high"], df["low"], df["close"], af0=config.PSAR_INIT_ACC, af=config.PSAR_ACC, max_af=config.PSAR_MAX_ACC)

        if psar is None or psar.empty:
            raise ValueError("PSAR calculation failed")

        psar_l = psar[f"PSARl_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        psar_s = psar[f"PSARs_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]

        psar_line = psar_l.fillna(psar_s)
        psar_dist = (df["close"] - psar_line) / df["close"]

        return pd.DataFrame({"psar_line": psar_line, "psar_dist": psar_dist}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        psar_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(psar_norm, columns=self.__cols, index=df.index)
