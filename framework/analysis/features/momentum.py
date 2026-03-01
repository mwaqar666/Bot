from .base import Feature
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Moving Average Convergence Divergence
# -----------------
class MovingAverageConvergenceDivergence(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        macd_data = ta.macd(df["close"], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)

        if macd_data is None or macd_data.empty:
            raise ValueError("MACD calculation failed")

        macd = macd_data[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_signal = macd_data[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_hist = macd_data[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]

        return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self._check_scaler()

        cols = ["macd_hist"]
        self._scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_scaler()

        cols = ["macd_hist"]
        norm = self._scaler.transform(df[cols])

        return pd.DataFrame(norm, columns=cols, index=df.index)


# -----------------
# 2. Relative Strength Index
# -----------------
class RelativeStrengthIndex(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df["close"], length=config.RSI_LENGTH)

        if rsi is None or rsi.empty:
            raise ValueError("RSI calculation failed")

        return pd.DataFrame({"rsi": rsi}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self._check_scaler()

        cols = ["rsi"]
        self._scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_scaler()

        cols = ["rsi"]
        norm = self._scaler.transform(df[cols])

        return pd.DataFrame(norm, columns=cols, index=df.index)


# -----------------
# 3. TTM Squeeze
# -----------------
class TTMSqueeze(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        squeeze = ta.squeeze(df["high"], df["low"], df["close"], bb_length=config.BBANDS_LENGTH, bb_std=config.BBANDS_STD, kc_length=config.SQUEEZE_KC_LENGTH, kc_scalar=config.SQUEEZE_KC_SCALAR, mom_length=config.SQUEEZE_MOM_LENGTH, mom_smooth=config.SQUEEZE_MOM_SMOOTH, mamode=config.SQUEEZE_MA_MODE)

        if squeeze is None or squeeze.empty:
            raise ValueError("TTM Squeeze calculation failed")

        sqz = squeeze[f"SQZ_{config.BBANDS_LENGTH}_{config.BBANDS_STD}_{config.SQUEEZE_KC_LENGTH}_{config.SQUEEZE_KC_SCALAR}"]
        sqz_on = squeeze["SQZ_ON"]
        sqz_off = squeeze["SQZ_OFF"]
        sqz_no = squeeze["SQZ_NO"]

        return pd.DataFrame({"sqz": sqz, "sqz_on": sqz_on, "sqz_off": sqz_off, "sqz_no": sqz_no}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        cols = ["sqz", "sqz_on"]
        self.__robust_scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["sqz", "sqz_on"]
        sqz = self.__robust_scaler.transform(df[cols]).clip(-5, 5)

        return pd.DataFrame(sqz, columns=cols, index=df.index)


# -----------------
# 4. Percentag Volume Oscillator
# -----------------
class PercentageVolumeOscillator(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        pvo_data = ta.pvo(df["volume"], fast=config.PVO_FAST, slow=config.PVO_SLOW, signal=config.PVO_SIGNAL)

        if pvo_data is None or pvo_data.empty:
            raise ValueError("PVO calculation failed")

        pvo = pvo_data[f"PVO_{config.PVO_FAST}_{config.PVO_SLOW}_{config.PVO_SIGNAL}"]
        pvo_hist = pvo_data[f"PVOh_{config.PVO_FAST}_{config.PVO_SLOW}_{config.PVO_SIGNAL}"]
        pvo_signal = pvo_data[f"PVOs_{config.PVO_FAST}_{config.PVO_SLOW}_{config.PVO_SIGNAL}"]

        return pd.DataFrame({"pvo": pvo, "pvo_hist": pvo_hist, "pvo_signal": pvo_signal}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        cols = ["pvo", "pvo_hist"]
        self.__robust_scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["pvo", "pvo_hist"]
        pvo = self.__robust_scaler.transform(df[cols]).clip(-5, 5)

        return pd.DataFrame(pvo, columns=cols, index=df.index)


# -----------------
# 5. Balance of Power
# -----------------
class BalanceOfPower(Feature):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        bop = ta.bop(df["open"], df["high"], df["low"], df["close"])

        if bop is None or bop.empty:
            raise ValueError("Balance of Power calculation failed")

        return pd.DataFrame({"bop": bop}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"bop": df["bop"].clip(-5, 5)}, index=df.index)


# -----------------
# 6. Williams %R
# -----------------
class WilliamsR(Feature):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        willr = ta.willr(df["high"], df["low"], df["close"], length=config.WILLR_LENGTH)

        if willr is None or willr.empty:
            raise ValueError("Williams %R calculation failed")

        return pd.DataFrame({"willr": willr}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        self.__robust_scaler.fit(df[["willr"]])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        willr = self.__robust_scaler.transform(df[["willr"]]).clip(-5, 5)
        return pd.DataFrame({"willr": willr.flatten()}, index=df.index)
