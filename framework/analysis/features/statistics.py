from .base import Indicator
from typing_extensions import Self

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

    def fit(self, df: pd.DataFrame) -> Self:
        self.__min_max_scaler.fit(df[["entropy"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        entropy = self.__min_max_scaler.transform(df[["entropy"]]).clip(-5, 5)
        return pd.DataFrame({"entropy": entropy.flatten()}, index=df.index)
