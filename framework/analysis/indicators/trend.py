from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
import numpy as np


# -----------------
# 1. ADX (Average Directional Index)
# -----------------
class ADX(Indicator):
    """
    Average Directional Index.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        adx = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_LENGTH)
        if adx is not None and not adx.empty:
            df["adx"] = adx[f"ADX_{config.ADX_LENGTH}"]
            df["dmp"] = adx[f"DMP_{config.ADX_LENGTH}"]
            df["dmn"] = adx[f"DMN_{config.ADX_LENGTH}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
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


# -----------------
# 2. Aroon
# -----------------
class Aroon(Indicator):
    """
    Aroon Indicator.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        aroon = ta.aroon(df["high"], df["low"], length=config.AROON_LENGTH, scalar=config.AROON_SCALAR)
        if aroon is not None and not aroon.empty:
            df["aroon_up"] = aroon[f"AROONU_{config.AROON_LENGTH}"]
            df["aroon_down"] = aroon[f"AROOND_{config.AROON_LENGTH}"]
            df["aroon_osc"] = aroon[f"AROONOSC_{config.AROON_LENGTH}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
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


# -----------------
# 3. Choppiness Index
# -----------------
class ChoppinessIndex(Indicator):
    """
    Choppiness Index.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        chop = ta.chop(df["high"], df["low"], df["close"], length=config.CHOP_LENGTH, atr_length=config.CHOP_ATR_LENGTH, ln=config.CHOP_LN, scalar=config.CHOP_SCALAR)
        if chop is not None and not chop.empty:
            df["chop"] = chop
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Filter: Low Chop = Subscribe to Trend Signals. High Chop = Ignore Trend Signals.
        return SignalDirection.NONE


# -----------------
# 4. Parabolic SAR
# -----------------
class ParabolicSAR(Indicator):
    """
    Parabolic Stop and Reverse.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        psar = ta.psar(df["high"], df["low"], df["close"], af0=config.PSAR_INIT_ACC, af=config.PSAR_ACC, max_af=config.PSAR_MAX_ACC)
        if psar is not None and not psar.empty:
            psar_l = psar[f"PSARl_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
            psar_s = psar[f"PSARs_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]

            df["psar"] = psar_l.fillna(psar_s)
            df["psar_direction"] = np.where(psar_l.notna(), 1, -1)
            df["psar_af"] = psar[f"PSARaf_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
            df["psar_r"] = psar[f"PSARr_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
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


# -----------------
# 5. Vortex Indicator
# -----------------
class Vortex(Indicator):
    """
    Vortex Indicator.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vortex = ta.vortex(df["high"], df["low"], df["close"], length=config.VORTEX_LENGTH)
        if vortex is not None and not vortex.empty:
            df["vortex_p"] = vortex[f"VTXP_{config.VORTEX_LENGTH}"]
            df["vortex_m"] = vortex[f"VTXM_{config.VORTEX_LENGTH}"]
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "vortex_p" not in df.columns:
            return SignalDirection.NONE

        vip = df["vortex_p"].iloc[current_idx]
        vim = df["vortex_m"].iloc[current_idx]

        if vip > vim:
            return SignalDirection.BUY
        elif vim > vip:
            return SignalDirection.SELL

        return SignalDirection.NONE
