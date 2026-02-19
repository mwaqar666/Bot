from .base import Indicator

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler


# -----------------
# 1. Entropy
# -----------------
class Entropy(Indicator):
    def __init__(self) -> None:
        self.__min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = ta.entropy(df["close"], length=config.ENTROPY_LENGTH, base=config.ENTROPY_BASE)

        if entropy is None or entropy.empty:
            raise ValueError("Entropy calculation failed")

        return pd.DataFrame({"entropy": entropy}, index=df.index)

    def fit_scaler(self, df: pd.DataFrame) -> None:
        self.__min_max_scaler.fit(df[["entropy"]])

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = self.__min_max_scaler.transform(df[["entropy"]])
        return pd.DataFrame({"entropy": entropy.flatten()}, index=df.index)
