from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Chaikin Money Flow
# -----------------
class ChaikinMoneyFlow(Indicator):
    """
    Chaikin Money Flow.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)

        if cmf is None or cmf.empty:
            raise ValueError("Chaikin Money Flow calculation failed")

        return pd.DataFrame({"cmf": cmf}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "cmf" not in df.columns:
            return SignalDirection.NONE

        val = df["cmf"].iloc[current_idx]

        # Zero Line Cross
        if val > 0.05:
            return SignalDirection.BUY
        elif val < -0.05:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = self.__robust_scaler.fit_transform(df[["cmf"]])
        return pd.DataFrame({"cmf": cmf.flatten()}, index=df.index)


# -----------------
# 2. Elder Force Index
# -----------------
class ElderForceIndex(Indicator):
    """
    Elder Force Index.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        efi = ta.efi(df["close"], df["volume"], length=config.EFI_LENGTH)

        if efi is None or efi.empty:
            raise ValueError("Elder Force Index calculation failed")

        return pd.DataFrame({"efi": efi}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
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

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        efi = self.__robust_scaler.fit_transform(df[["efi"]])
        return pd.DataFrame({"efi": efi.flatten()}, index=df.index)


# -----------------
# 3. Money Flow Index
# -----------------
class MoneyFlowIndex(Indicator):
    """
    Money Flow Index.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)

        if mfi is None or mfi.empty:
            raise ValueError("Money Flow Index calculation failed")

        return pd.DataFrame({"mfi": mfi}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "mfi" not in df.columns:
            return SignalDirection.NONE

        val = df["mfi"].iloc[current_idx]

        # Overbought/Oversold
        if val < 20:
            return SignalDirection.BUY
        elif val > 80:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = self.__min_max_scaler.fit_transform(df[["mfi"]])
        return pd.DataFrame({"mfi": mfi.flatten()}, index=df.index)


# -----------------
# 4. On-Balance Volume (OBV)
# -----------------
class OnBalanceVolume(Indicator):
    """
    On-Balance Volume.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df["close"], df["volume"])

        if obv is None or obv.empty:
            raise ValueError("On-Balance Volume calculation failed")

        return pd.DataFrame({"obv": obv}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # OBV is primarily for divergence or trend confirmation.
        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        obv_diff = df["obv"].diff().fillna(0)
        obv = self.__robust_scaler.fit_transform(obv_diff.values.reshape(-1, 1))
        return pd.DataFrame({"obv": obv.flatten()}, index=df.index)


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

        return pd.DataFrame({}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        return SignalDirection.NONE
