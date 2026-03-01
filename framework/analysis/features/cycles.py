from .base import Indicator
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. Even Better Sine Wave
# -----------------
class EvenBetterSineWave(Indicator):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ebsw = ta.ebsw(df["close"], length=config.EBSW_LENGTH, bars=config.EBSW_BARS)

        if ebsw is None or ebsw.empty:
            raise ValueError("Even Better Sine Wave calculation failed")

        return pd.DataFrame({"ebsw": ebsw}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["ebsw"]
        return pd.DataFrame(df[cols], columns=cols, index=df.index)
