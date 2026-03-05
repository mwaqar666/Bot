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
        self.__scaler.fit(df[["adx"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self.__scaler.transform(df[["adx"]])
        return pd.DataFrame({"adx": scaled.flatten()}, index=df.index)


# -----------------
# 2. Parabolic Stop and Reverse
# -----------------
class ParabolicStopAndReverse(Feature):
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        psar = ta.psar(df["high"], df["low"], df["close"], af0=config.PSAR_INIT_ACC, af=config.PSAR_ACC, max_af=config.PSAR_MAX_ACC)

        if psar is None or psar.empty:
            raise ValueError("PSAR calculation failed")

        psar_l = psar[f"PSARl_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        psar_s = psar[f"PSARs_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]

        psar_line = psar_l.fillna(psar_s)
        psar_direction = np.where(psar_l.notna(), 1, -1)
        psar_dist = (df["close"] - psar_line) / df["close"]

        return pd.DataFrame(
            {
                "psar": psar_line,
                "psar_direction": psar_direction,
                "psar_dist": psar_dist,
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["psar_dist"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        psar_dist = self.__scaler.transform(df[["psar_dist"]])

        return pd.DataFrame(
            {
                "psar_dist": psar_dist.flatten(),
                "psar_direction": df["psar_direction"].astype(float),
            },
            index=df.index,
        )
