from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


# -----------------
# 1. Chaikin Money Flow
# -----------------
class ChaikinMoneyFlow(Feature):
    __cols = ["cmf"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)

        if cmf is None or cmf.empty:
            raise ValueError("Chaikin Money Flow calculation failed")

        return pd.DataFrame({"cmf": cmf}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(cmf_norm, columns=self.__cols, index=df.index)


# -----------------
# 2. Money Flow Index
# -----------------
class MoneyFlowIndex(Feature):
    __cols = ["mfi"]

    def __init__(self) -> None:
        self.__scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)

        if mfi is None or mfi.empty:
            raise ValueError("Money Flow Index calculation failed")

        return pd.DataFrame({"mfi": mfi}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(mfi_norm, columns=self.__cols, index=df.index)


# -----------------
# 3. Volume Ratio
# -----------------
class VolumeRatio(Feature):
    __cols = ["volume_ratio"]

    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = df["volume"].rolling(window=20, min_periods=1).mean()
        volume_ratio = df["volume"] / rolling_mean
        return pd.DataFrame({"volume_ratio": volume_ratio}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        volume_ratio_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(volume_ratio_norm, columns=self.__cols, index=df.index)
