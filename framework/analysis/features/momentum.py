from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


# -----------------
# 1. Moving Average Convergence Divergence
# -----------------
class MovingAverageConvergenceDivergence(Feature):
    __cols = ["macd_hist"]

    def __init__(self):
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = ta.macd(df["close"], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)

        if macd is None or macd.empty:
            raise ValueError("MACD calculation failed")

        macd_line = macd[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_signal = macd[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_hist = macd[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_hist = macd_hist / df["close"]

        return pd.DataFrame({"macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        macd_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(macd_norm, columns=self.__cols, index=df.index)


# -----------------
# 2. Relative Strength Index
# -----------------
class RelativeStrengthIndex(Feature):
    __cols = ["rsi"]

    def __init__(self) -> None:
        self.__scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df["close"], length=config.RSI_LENGTH)

        if rsi is None or rsi.empty:
            raise ValueError("RSI calculation failed")

        return pd.DataFrame({"rsi": rsi}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[self.__cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi_norm = self.__scaler.transform(df[self.__cols])
        return pd.DataFrame(rsi_norm, columns=self.__cols, index=df.index)


# -----------------
# 3. TTM Squeeze
# -----------------
class TTMSqueeze(Feature):
    def __init__(self) -> None:
        self.__scaler = QuantileTransformer(output_distribution="normal")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        sqz_data = ta.squeeze(df["high"], df["low"], df["close"], bb_length=config.BBANDS_LENGTH, bb_std=config.BBANDS_STD, kc_length=config.SQUEEZE_KC_LENGTH, kc_scalar=config.SQUEEZE_KC_SCALAR, mom_length=config.SQUEEZE_MOM_LENGTH, mom_smooth=config.SQUEEZE_MOM_SMOOTH, mamode=config.SQUEEZE_MA_MODE)

        if sqz_data is None or sqz_data.empty:
            raise ValueError("TTM Squeeze calculation failed")

        sqz = sqz_data[f"SQZ_{config.BBANDS_LENGTH}_{config.BBANDS_STD}_{config.SQUEEZE_KC_LENGTH}_{config.SQUEEZE_KC_SCALAR}"]
        sqz_on = sqz_data["SQZ_ON"]

        sqz = sqz / df["close"]

        return pd.DataFrame({"sqz": sqz, "sqz_on": sqz_on}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__scaler.fit(df[["sqz"]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sqz": self.__scaler.transform(df[["sqz"]]).flatten(),
                "sqz_on": df["sqz_on"].astype(float),  # binary: 1 = squeeze active, 0 = not
            },
            index=df.index,
        )
