from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Arnaud Legoux Moving Average
# -----------------
class ArnaudLegouxMovingAverage(Indicator):
    """
    Arnaud Legoux Moving Average.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        alma = ta.alma(df["close"], length=config.ALMA_LENGTH, sigma=config.ALMA_SIGMA, distribution_offset=config.ALMA_DISTRIBUTION_OFFSET)

        if alma is None or alma.empty:
            raise ValueError("ALMA calculation failed")

        return pd.DataFrame({"alma": alma}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "alma" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        alma = df["alma"].iloc[current_idx]

        # Basic Price-MA Logic
        if price > alma:
            return SignalDirection.BUY
        elif price < alma:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["alma"]) / df["close"]
        alma = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"alma": alma.flatten()}, index=df.index)


# -----------------
# 2. Exponential Moving Average
# -----------------
class ExponentialMovingAverage(Indicator):
    """
    Exponential Moving Average.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ema = ta.ema(df["close"], length=config.EMA_LENGTH)

        if ema is None or ema.empty:
            raise ValueError("EMA calculation failed")

        return pd.DataFrame({"ema": ema}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "ema" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        ema = df["ema"].iloc[current_idx]

        if price > ema:
            return SignalDirection.BUY
        elif price < ema:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["ema"]) / df["close"]
        ema = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"ema": ema.flatten()}, index=df.index)


# -----------------
# 3. Hull Moving Average
# -----------------
class HullMovingAverage(Indicator):
    """
    Hull Moving Average.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        hma = ta.hma(df["close"], length=config.HMA_LENGTH)

        if hma is None or hma.empty:
            raise ValueError("HMA calculation failed")

        return pd.DataFrame({"hma": hma}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "hma" not in df.columns:
            return SignalDirection.NONE

        curr_hma = df["hma"].iloc[current_idx]
        prev_hma = df["hma"].iloc[current_idx - 1]

        # HMA Turning Points
        if curr_hma > prev_hma:
            return SignalDirection.BUY
        elif curr_hma < prev_hma:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["hma"]) / df["close"]
        hma = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"hma": hma.flatten()}, index=df.index)


# -----------------
# 4. Super Trend
# -----------------
class SuperTrend(Indicator):
    """
    Super Trend.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)

        if super_trend is None or super_trend.empty:
            raise ValueError("SuperTrend calculation failed")

        st = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_direction = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]

        return pd.DataFrame({"st": st, "st_direction": st_direction}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        """
        SuperTrend Direction (1 = Uptrend, -1 = Downtrend)
        """
        if "st_direction" not in df.columns:
            return SignalDirection.NONE

        direction = df["st_direction"].iloc[current_idx]

        if direction == 1:
            return SignalDirection.BUY
        elif direction == -1:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["st"]) / df["close"]
        st = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"st": st.flatten(), "st_direction": df["st_direction"]}, index=df.index)


# -----------------
# 5. Volume Weighted Average Price
# -----------------
class VolumeWeightedAveragePrice(Indicator):
    """
    Volume Weighted Average Price.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        if vwap is None or vwap.empty:
            raise ValueError("VWAP calculation failed")

        return pd.DataFrame({"vwap": vwap}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "vwap" not in df.columns:
            return SignalDirection.NONE

        price = df["close"].iloc[current_idx]
        vwap = df["vwap"].iloc[current_idx]

        if price > vwap:
            return SignalDirection.BUY
        elif price < vwap:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        vwap = self.__robust_scaler.fit_transform(percentage_distance.values.reshape(-1, 1))
        return pd.DataFrame({"vwap": vwap.flatten()}, index=df.index)
