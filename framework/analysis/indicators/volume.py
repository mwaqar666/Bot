from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta


# -----------------
# 1. Chaikin Money Flow (CMF)
# -----------------
class CMF(Indicator):
    """
    Chaikin Money Flow.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)
        if cmf is not None and not cmf.empty:
            df["cmf"] = cmf
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "cmf" not in df.columns:
            return SignalDirection.NONE

        val = df["cmf"].iloc[current_idx]

        # Zero Line Cross
        if val > 0.05:
            return SignalDirection.BUY
        elif val < -0.05:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 2. Elder Force Index (EFI)
# -----------------
class EFI(Indicator):
    """
    Elder Force Index.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        efi = ta.efi(df["close"], df["volume"], length=config.EFI_LENGTH)
        if efi is not None and not efi.empty:
            df["efi"] = efi
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "efi" not in df.columns:
            return SignalDirection.NONE

        val = df["efi"].iloc[current_idx]
        prev = df["efi"].iloc[current_idx - 1]

        # Zero Cross
        if prev < 0 and val > 0:
            return SignalDirection.BUY
        elif prev > 0 and val < 0:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 3. Money Flow Index (MFI)
# -----------------
class MFI(Indicator):
    """
    Money Flow Index.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)
        if mfi is not None and not mfi.empty:
            df["mfi"] = mfi
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "mfi" not in df.columns:
            return SignalDirection.NONE

        val = df["mfi"].iloc[current_idx]

        # Overbought/Oversold
        if val < 20:
            return SignalDirection.BUY
        elif val > 80:
            return SignalDirection.SELL

        return SignalDirection.NONE


# -----------------
# 4. On-Balance Volume (OBV)
# -----------------
class OBV(Indicator):
    """
    On-Balance Volume.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df["close"], df["volume"])
        if obv is not None and not obv.empty:
            df["obv"] = obv
        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # OBV is primarily for divergence or trend confirmation.
        return SignalDirection.NONE


# -----------------
# 5. Volume Profile (VP)
# -----------------
class VolumeProfile(Indicator):
    """
    Volume Profile.
    Note: Returns price levels, hard to map to time series directly without logic.
    For now, we keep it disabled or experimental.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # VP calculation logic in pandas_ta returns a DF with different index (price buckets)
        # So we cannot easily attach it to the main DF without complex logic.
        # Check if enabled in config or logic.

        # Example dummy implementation to match structure
        # vp = ta.vp(df["close"], df["volume"], width=config.VP_LENGTH)
        # if vp is not None and not vp.empty:
        #     df["vp_low"] = vp["low_close"]
        #     df["vp_mean"] = vp["mean_close"]
        #     df["vp_high"] = vp["high_close"]
        #     df["vp_pos"] = vp["pos_volume"]
        #     df["vp_neg"] = vp["neg_volume"]
        #     df["vp_total"] = vp["total_volume"]

        return df

    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        return SignalDirection.NONE
