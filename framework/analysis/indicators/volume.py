from .base import Indicator

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Chaikin Money Flow
# -----------------
class ChaikinMoneyFlow(Indicator):
    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)

        if cmf is None or cmf.empty:
            raise ValueError("Chaikin Money Flow calculation failed")

        return pd.DataFrame({"cmf": cmf}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cmf = self.__robust_scaler.fit_transform(df[["cmf"]])
        return pd.DataFrame({"cmf": cmf.flatten()}, index=df.index)


# -----------------
# 2. Elder Force Index
# -----------------
class ElderForceIndex(Indicator):
    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        efi = ta.efi(df["close"], df["volume"], length=config.EFI_LENGTH)

        if efi is None or efi.empty:
            raise ValueError("Elder Force Index calculation failed")

        return pd.DataFrame({"efi": efi}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        efi = self.__robust_scaler.fit_transform(df[["efi"]])
        return pd.DataFrame({"efi": efi.flatten()}, index=df.index)


# -----------------
# 3. Money Flow Index
# -----------------
class MoneyFlowIndex(Indicator):
    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)

        if mfi is None or mfi.empty:
            raise ValueError("Money Flow Index calculation failed")

        return pd.DataFrame({"mfi": mfi}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        mfi = self.__min_max_scaler.fit_transform(df[["mfi"]])
        return pd.DataFrame({"mfi": mfi.flatten()}, index=df.index)


# -----------------
# 4. On-Balance Volume (OBV)
# -----------------
class OnBalanceVolume(Indicator):
    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df["close"], df["volume"])

        if obv is None or obv.empty:
            raise ValueError("On-Balance Volume calculation failed")

        return pd.DataFrame({"obv": obv}, index=df.index)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        obv_diff = df["obv"].diff().fillna(0)
        obv = self.__robust_scaler.fit_transform(obv_diff.values.reshape(-1, 1))
        return pd.DataFrame({"obv": obv.flatten()}, index=df.index)


# -----------------
# 5. Volume Profile (VP)
# -----------------
class VolumeProfile(Indicator):
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

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({}, index=df.index)
