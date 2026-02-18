from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Average Directional Index
# -----------------
class AverageDirectionalIndex(Indicator):
    """
    Average Directional Index.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_data = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_LENGTH)

        if adx_data is None or adx_data.empty:
            raise ValueError("ADX calculation failed")

        adx = adx_data[f"ADX_{config.ADX_LENGTH}"]
        dmp = adx_data[f"DMP_{config.ADX_LENGTH}"]
        dmn = adx_data[f"DMN_{config.ADX_LENGTH}"]

        return pd.DataFrame({"adx": adx, "dmp": dmp, "dmn": dmn}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "adx" not in df.columns:
            return SignalDirection.NONE

        adx_val = df["adx"].iloc[current_idx]
        dmp = df["dmp"].iloc[current_idx]
        dmn = df["dmn"].iloc[current_idx]

        # Trend Strength Filter
        if adx_val > 25:
            if dmp > dmn:
                return SignalDirection.BUY
            elif dmn > dmp:
                return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["adx", "dmp", "dmn"]
        adx = self.__min_max_scaler.fit_transform(df[cols])
        return pd.DataFrame(adx, columns=cols, index=df.index)


# -----------------
# 2. Aroon Oscillator
# -----------------
class AroonOscillator(Indicator):
    """
    Aroon Oscillator.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        aroon = ta.aroon(df["high"], df["low"], length=config.AROON_LENGTH, scalar=config.AROON_SCALAR)

        if aroon is None or aroon.empty:
            raise ValueError("Aroon calculation failed")

        aroon_up = aroon[f"AROONU_{config.AROON_LENGTH}"]
        aroon_down = aroon[f"AROOND_{config.AROON_LENGTH}"]
        aroon_osc = aroon[f"AROONOSC_{config.AROON_LENGTH}"]

        return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down, "aroon_osc": aroon_osc}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "aroon_up" not in df.columns:
            return SignalDirection.NONE

        up = df["aroon_up"].iloc[current_idx]
        down = df["aroon_down"].iloc[current_idx]

        # Crossover Logic
        if up > down and up > 70:
            return SignalDirection.BUY
        elif down > up and down > 70:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["aroon_up", "aroon_down", "aroon_osc"]
        aroon = self.__min_max_scaler.fit_transform(df[cols])
        return pd.DataFrame(aroon, columns=cols, index=df.index)


# -----------------
# 3. Choppiness Index
# -----------------
class ChoppinessIndex(Indicator):
    """
    Choppiness Index.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        chop = ta.chop(df["high"], df["low"], df["close"], length=config.CHOP_LENGTH, atr_length=config.CHOP_ATR_LENGTH, ln=config.CHOP_LN, scalar=config.CHOP_SCALAR)

        if chop is None or chop.empty:
            raise ValueError("Chop calculation failed")

        return pd.DataFrame({"chop": chop}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Filter: Low Chop = Subscribe to Trend Signals. High Chop = Ignore Trend Signals.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        chop = self.__min_max_scaler.fit_transform(df[["chop"]])
        return pd.DataFrame({"chop": chop.flatten()}, index=df.index)


# -----------------
# 4. Parabolic Stop and Reverse
# -----------------
class ParabolicStopAndReverse(Indicator):
    """
    Parabolic Stop and Reverse.
    """

    __robust_scaler = RobustScaler()
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        psar = ta.psar(df["high"], df["low"], df["close"], af0=config.PSAR_INIT_ACC, af=config.PSAR_ACC, max_af=config.PSAR_MAX_ACC)

        if psar is None or psar.empty:
            raise ValueError("PSAR calculation failed")

        psar_l = psar[f"PSARl_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        psar_s = psar[f"PSARs_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        psar_af = psar[f"PSARaf_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        psar_r = psar[f"PSARr_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]

        psar = psar_l.fillna(psar_s)
        psar_direction = np.where(psar_l.notna(), 1, -1)

        return pd.DataFrame({"psar": psar, "psar_direction": psar_direction, "psar_af": psar_af, "psar_r": psar_r}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "psar_direction" not in df.columns:
            return SignalDirection.NONE

        direction = df["psar_direction"].iloc[current_idx]
        prev_direction = df["psar_direction"].iloc[current_idx - 1]

        # Reversal Signal
        if prev_direction == -1 and direction == 1:
            return SignalDirection.BUY
        elif prev_direction == 1 and direction == -1:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["psar"]) / df["close"]
        psar = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))

        psar_af = self.__min_max_scaler.fit_transform(df[["psar_af"]])

        return pd.DataFrame({"psar": psar.flatten(), "psar_direction": df["psar_direction"], "psar_af": psar_af.flatten()}, index=df.index)


# -----------------
# 5. Vortex Indicator
# -----------------
class Vortex(Indicator):
    """
    Vortex Indicator.
    """

    __scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vortex = ta.vortex(df["high"], df["low"], df["close"], length=config.VORTEX_LENGTH)

        if vortex is None or vortex.empty:
            raise ValueError("Vortex calculation failed")

        vortex_p = vortex[f"VTXP_{config.VORTEX_LENGTH}"]
        vortex_m = vortex[f"VTXM_{config.VORTEX_LENGTH}"]
        vortex_osc = vortex_p - vortex_m

        return pd.DataFrame({"vortex_p": vortex_p, "vortex_m": vortex_m, "vortex_osc": vortex_osc}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "vortex_p" not in df.columns:
            return SignalDirection.NONE

        vip = df["vortex_p"].iloc[current_idx]
        vim = df["vortex_m"].iloc[current_idx]

        if vip > vim:
            return SignalDirection.BUY
        elif vim > vip:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["vortex_p", "vortex_m", "vortex_osc"]
        vortex = self.__scaler.fit_transform(df[cols])
        return pd.DataFrame(vortex, columns=cols, index=df.index)
