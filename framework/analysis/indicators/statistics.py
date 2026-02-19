from .base import Indicator

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Entropy
# -----------------
class Entropy(Indicator):
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = ta.entropy(df["close"], length=config.ENTROPY_LENGTH, base=config.ENTROPY_BASE)

        if entropy is None or entropy.empty:
            raise ValueError("Entropy calculation failed")

        return pd.DataFrame({"entropy": entropy}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = self.__min_max_scaler.fit_transform(df[["entropy"]])
        return pd.DataFrame({"entropy": entropy.flatten()}, index=df.index)


# -----------------
# 2. Z-Score
# -----------------
class ZScore(Indicator):
    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        zscore = ta.zscore(df["close"], length=config.ZSCORE_LENGTH, std=config.ZSCORE_STD)

        if zscore is None or zscore.empty:
            raise ValueError("Z-score calculation failed")

        return pd.DataFrame({"zscore": zscore}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        zscore = self.__robust_scaler.fit_transform(df[["zscore"]])
        return pd.DataFrame({"zscore": zscore.flatten()}, index=df.index)
