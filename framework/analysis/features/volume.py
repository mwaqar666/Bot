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
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)

        if cmf is None or cmf.empty:
            raise ValueError("Chaikin Money Flow calculation failed")

        return pd.DataFrame({"cmf": cmf}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["cmf"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = self.__scaler.transform(df[["cmf"]])
        return pd.DataFrame({"cmf": cmf.flatten()}, index=df.index)


# -----------------
# 2. Money Flow Index
# -----------------
class MoneyFlowIndex(Feature):
    def __init__(self) -> None:
        self.__minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.__quantile_scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)

        if mfi is None or mfi.empty:
            raise ValueError("Money Flow Index calculation failed")

        return pd.DataFrame({"mfi": mfi, "mfi_diff": mfi.diff()}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__minmax_scaler.fit(df[["mfi"]])
        self.__quantile_scaler.fit(df[["mfi_diff"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = self.__minmax_scaler.transform(df[["mfi"]])
        mfi_diff = self.__quantile_scaler.transform(df[["mfi_diff"]])

        return pd.DataFrame(
            {"mfi": mfi.flatten(), "mfi_diff": mfi_diff.flatten()},
            index=df.index,
        )


# -----------------
# 3. Volume Ratio
# -----------------
class VolumeRatio(Feature):
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = df["volume"].rolling(window=20, min_periods=1).mean()
        volume_ratio = df["volume"] / rolling_mean
        return pd.DataFrame({"volume_ratio": volume_ratio}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["volume_ratio"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        volume_ratio = self.__scaler.transform(df[["volume_ratio"]])
        return pd.DataFrame({"volume_ratio": volume_ratio.flatten()}, index=df.index)
