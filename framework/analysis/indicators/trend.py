from .base import Indicator

import config

import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Average Directional Index
# -----------------
class AverageDirectionalIndex(Indicator):
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_data = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_LENGTH)

        if adx_data is None or adx_data.empty:
            raise ValueError("ADX calculation failed")

        adx = adx_data[f"ADX_{config.ADX_LENGTH}"]
        dmp = adx_data[f"DMP_{config.ADX_LENGTH}"]
        dmn = adx_data[f"DMN_{config.ADX_LENGTH}"]

        return pd.DataFrame({"adx": adx, "dmp": dmp, "dmn": dmn}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["adx", "dmp", "dmn"]
        adx = self.__min_max_scaler.fit_transform(df[cols])
        return pd.DataFrame(adx, columns=cols, index=df.index)


# -----------------
# 2. Aroon Oscillator
# -----------------
class AroonOscillator(Indicator):
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        aroon_data = ta.aroon(df["high"], df["low"], length=config.AROON_LENGTH, scalar=config.AROON_SCALAR)

        if aroon_data is None or aroon_data.empty:
            raise ValueError("Aroon calculation failed")

        aroon = aroon_data[f"AROONOSC_{config.AROON_LENGTH}"]
        aroon_up = aroon_data[f"AROONU_{config.AROON_LENGTH}"]
        aroon_down = aroon_data[f"AROOND_{config.AROON_LENGTH}"]

        return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down, "aroon": aroon}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        aroon = self.__min_max_scaler.fit_transform(df[["aroon"]])

        return pd.DataFrame({"aroon": aroon.flatten()}, index=df.index)


# -----------------
# 3. Choppiness Index
# -----------------
class ChoppinessIndex(Indicator):
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        chop = ta.chop(df["high"], df["low"], df["close"], length=config.CHOP_LENGTH, atr_length=config.CHOP_ATR_LENGTH, ln=config.CHOP_LN, scalar=config.CHOP_SCALAR)

        if chop is None or chop.empty:
            raise ValueError("Chop calculation failed")

        return pd.DataFrame({"chop": chop}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        chop = self.__min_max_scaler.fit_transform(df[["chop"]])
        return pd.DataFrame({"chop": chop.flatten()}, index=df.index)


# -----------------
# 4. Parabolic Stop and Reverse
# -----------------
class ParabolicStopAndReverse(Indicator):
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

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["psar"]) / df["close"]
        psar = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))

        psar_af = self.__min_max_scaler.fit_transform(df[["psar_af"]])

        return pd.DataFrame({"psar": psar.flatten(), "psar_direction": df["psar_direction"], "psar_af": psar_af.flatten()}, index=df.index)


# -----------------
# 5. Vortex Indicator
# -----------------
class Vortex(Indicator):
    __scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vortex_data = ta.vortex(df["high"], df["low"], df["close"], length=config.VORTEX_LENGTH)

        if vortex_data is None or vortex_data.empty:
            raise ValueError("Vortex calculation failed")

        vortex_p = vortex_data[f"VTXP_{config.VORTEX_LENGTH}"]
        vortex_m = vortex_data[f"VTXM_{config.VORTEX_LENGTH}"]
        vortex = vortex_p - vortex_m

        return pd.DataFrame({"vortex_p": vortex_p, "vortex_m": vortex_m, "vortex": vortex}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        vortex = self.__scaler.fit_transform(df[["vortex"]])

        return pd.DataFrame({"vortex": vortex.flatten()}, index=df.index)
