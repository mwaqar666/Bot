from .base import Indicator
from typing_extensions import Self

import config

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import RobustScaler


# -----------------
# 1. Super Trend
# -----------------
class SuperTrend(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)

        if super_trend is None or super_trend.empty:
            raise ValueError("SuperTrend calculation failed")

        st = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
        st_direction = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]

        return pd.DataFrame({"st": st, "st_direction": st_direction}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        percentage_distance = (df["close"] - df["st"]) / df["close"]
        self.__robust_scaler.fit(percentage_distance.values.reshape(-1, 1))
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["st"]) / df["close"]
        st = self.__robust_scaler.transform(percentage_distance.values.reshape(-1, 1)).clip(-5, 5)
        return pd.DataFrame({"st": st.flatten(), "st_direction": df["st_direction"]}, index=df.index)


# -----------------
# 2. Volume Weighted Average Price
# -----------------
class VolumeWeightedAveragePrice(Indicator):
    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        if vwap is None or vwap.empty:
            raise ValueError("VWAP calculation failed")

        return pd.DataFrame({"vwap": vwap}, index=df.index)

    def fit(self, df: pd.DataFrame) -> Self:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        self.__robust_scaler.fit(percentage_distance.values.reshape(-1, 1))
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        percentage_distance = (df["close"] - df["vwap"]) / df["close"]
        vwap = self.__robust_scaler.transform(percentage_distance.values.reshape(-1, 1)).clip(-5, 5)
        return pd.DataFrame({"vwap": vwap.flatten()}, index=df.index)


# -----------------
# 3. Exponential Moving Average
# -----------------
class ExponentialMovingAverage(Indicator):
    _COLS: list[str] = [
        "ema_fast_diff",
        "ema_slow_diff",
        "ema_spread",
        "ema_spread_slope",
        "ema_fast_slope",
        "ema_slow_slope",
    ]

    def __init__(self) -> None:
        self.__robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 6 normalised EMA features from fast and slow EMAs.

        Args:
            df (pd.DataFrame): OHLCV DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 6 EMA-derived feature columns.
        """
        ema_fast = ta.ema(df["close"], length=config.EMA_FAST_LENGTH)
        ema_slow = ta.ema(df["close"], length=config.EMA_SLOW_LENGTH)

        if ema_fast is None or ema_fast.empty or ema_slow is None or ema_slow.empty:
            raise ValueError("EMA calculation failed")

        # Relative distance of price from each EMA (scale-invariant)
        ema_fast_diff = (df["close"] - ema_fast) / df["close"]
        ema_slow_diff = (df["close"] - ema_slow) / df["close"]

        # Relative spread between EMAs: crossover signal
        ema_spread = (ema_fast - ema_slow) / ema_slow

        # Slope of spread: is the crossover accelerating or decelerating?
        ema_spread_slope = ema_spread.diff()

        # Individual EMA slopes as % change per candle (already scale-invariant)
        ema_fast_slope = ema_fast.pct_change()
        ema_slow_slope = ema_slow.pct_change()

        return pd.DataFrame(
            {
                "ema_fast_diff": ema_fast_diff,
                "ema_slow_diff": ema_slow_diff,
                "ema_spread": ema_spread,
                "ema_spread_slope": ema_spread_slope,
                "ema_fast_slope": ema_fast_slope,
                "ema_slow_slope": ema_slow_slope,
            },
            index=df.index,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        cols = ["ema_fast_diff", "ema_slow_diff", "ema_spread", "ema_spread_slope", "ema_fast_slope", "ema_slow_slope"]
        self.__robust_scaler.fit(df[cols])
        return self

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["ema_fast_diff", "ema_slow_diff", "ema_spread", "ema_spread_slope", "ema_fast_slope", "ema_slow_slope"]
        scaled = self.__robust_scaler.transform(df[cols]).clip(-5, 5)
        return pd.DataFrame(scaled, columns=cols, index=df.index)
