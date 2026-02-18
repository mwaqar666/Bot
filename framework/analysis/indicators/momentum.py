from .base import Indicator

import config
from framework.data.data_types import SignalDirection

import pandas as pd
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# -----------------
# 1. Awesome Oscillator
# -----------------
class AwesomeOscillator(Indicator):
    """
    Awesome Oscillator: Market momentum.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ao = ta.ao(df["high"], df["low"], fast=config.AO_FAST, slow=config.AO_SLOW)

        if ao is None or ao.empty:
            raise ValueError("Awesome Oscillator calculation failed")

        return pd.DataFrame({"ao": ao}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        # Simple Zero Line Cross Logic (can be expanded to "Saucer")
        if "ao" not in df.columns:
            return SignalDirection.NONE

        current_ao = df["ao"].iloc[current_idx]
        prev_ao = df["ao"].iloc[current_idx - 1]

        # Bullish: Crosses Above Zero
        if prev_ao < 0 and current_ao > 0:
            return SignalDirection.BUY

        # Bearish: Crosses Below Zero
        if prev_ao > 0 and current_ao < 0:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        ao = self.__robust_scaler.fit_transform(df[["ao"]])
        return pd.DataFrame({"ao": ao.flatten()}, index=df.index)


# -----------------
# 2. Moving Average Convergence Divergence
# -----------------
class MovingAverageConvergenceDivergence(Indicator):
    """
    Moving Average Convergence Divergence.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        macd_data = ta.macd(df["close"], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)

        if macd_data is None or macd_data.empty:
            raise ValueError("MACD calculation failed")

        macd = macd_data[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_signal = macd_data[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
        macd_hist = macd_data[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]

        return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            return SignalDirection.NONE

        curr_macd = df["macd"].iloc[current_idx]
        curr_sig = df["macd_signal"].iloc[current_idx]
        prev_macd = df["macd"].iloc[current_idx - 1]
        prev_sig = df["macd_signal"].iloc[current_idx - 1]

        # Bullish Crossover
        if prev_macd < prev_sig and curr_macd > curr_sig:
            return SignalDirection.BUY

        # Bearish Crossover
        if prev_macd > prev_sig and curr_macd < curr_sig:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["macd", "macd_signal", "macd_hist"]
        macd = self.__robust_scaler.fit_transform(df[cols])

        return pd.DataFrame(macd, columns=cols, index=df.index)


# -----------------
# 3. Relative Strength Index
# -----------------
class RelativeStrengthIndex(Indicator):
    """
    Relative Strength Index.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df["close"], length=config.RSI_LENGTH)

        if rsi is None or rsi.empty:
            raise ValueError("RSI calculation failed")

        return pd.DataFrame({"rsi": rsi}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "rsi" not in df.columns:
            return SignalDirection.NONE

        val = df["rsi"].iloc[current_idx]

        # Simple Oversold/Overbought Logic
        if val < config.RSI_OVERSOLD:
            return SignalDirection.BUY
        elif val > config.RSI_OVERBOUGHT:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = self.__min_max_scaler.fit_transform(df[["rsi"]])
        return pd.DataFrame({"rsi": rsi.flatten()}, index=df.index)


# -----------------
# 4. Stochastic RSI
# -----------------
class StochasticRSI(Indicator):
    """
    Stochastic RSI.
    """

    __min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        stochrsi = ta.stochrsi(df["close"], length=config.STOCHRSI_LENGTH, rsi_length=config.STOCHRSI_RSI_LENGTH, k=config.STOCHRSI_K, d=config.STOCHRSI_D)

        if stochrsi is None or stochrsi.empty:
            raise ValueError("Stochastic RSI calculation failed")

        stochrsi_k = stochrsi[f"STOCHRSIk_{config.STOCHRSI_LENGTH}_{config.STOCHRSI_RSI_LENGTH}_{config.STOCHRSI_K}_{config.STOCHRSI_D}"]
        stochrsi_d = stochrsi[f"STOCHRSId_{config.STOCHRSI_LENGTH}_{config.STOCHRSI_RSI_LENGTH}_{config.STOCHRSI_K}_{config.STOCHRSI_D}"]

        return pd.DataFrame({"stochrsi_k": stochrsi_k, "stochrsi_d": stochrsi_d}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        if "stochrsi_k" not in df.columns or "stochrsi_d" not in df.columns:
            return SignalDirection.NONE

        k = df["stochrsi_k"].iloc[current_idx]
        d = df["stochrsi_d"].iloc[current_idx]
        prev_k = df["stochrsi_k"].iloc[current_idx - 1]
        prev_d = df["stochrsi_d"].iloc[current_idx - 1]

        # Bullish Crossover in Oversold Region (< 20)
        if k < 20 and d < 20 and prev_k < prev_d and k > d:
            return SignalDirection.BUY

        # Bearish Crossover in Overbought Region (> 80)
        if k > 80 and d > 80 and prev_k > prev_d and k < d:
            return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["stochrsi_k", "stochrsi_d"]
        stochrsi = self.__min_max_scaler.fit_transform(df[cols])

        return pd.DataFrame(stochrsi, columns=cols, index=df.index)


# -----------------
# 5. TTM Squeeze
# -----------------
class TTMSqueeze(Indicator):
    """
    TTM Squeeze.
    """

    __robust_scaler = RobustScaler()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        squeeze = ta.squeeze(df["high"], df["low"], df["close"], bb_length=config.BBANDS, bb_std=config.BBANDS_STD, kc_length=config.SQUEEZE_KC_LENGTH, kc_scalar=config.SQUEEZE_KC_SCALAR, mom_length=config.SQUEEZE_MOM_LENGTH, mom_smooth=config.SQUEEZE_MOM_SMOOTH, mamode=config.SQUEEZE_MA_MODE)

        if squeeze is None or squeeze.empty:
            raise ValueError("TTM Squeeze calculation failed")

        sqz = squeeze[f"SQZ_{config.BBANDS}_{config.BBANDS_STD}_{config.SQUEEZE_KC_LENGTH}_{config.SQUEEZE_KC_SCALAR}"]
        sqz_on = squeeze["SQZ_ON"]
        sqz_off = squeeze["SQZ_OFF"]
        sqz_no = squeeze["SQZ_NO"]

        return pd.DataFrame({"sqz": sqz, "sqz_on": sqz_on, "sqz_off": sqz_off, "sqz_no": sqz_no}, index=df.index)

    def signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        """
        TTM Squeeze Signal:
        - We look for a 'Squeeze Fire' (SQZ_OFF turns 1 after being 0).
        - Direction is determined by the momentum histogram (sqz).
        """
        if "sqz_off" not in df.columns or "sqz" not in df.columns:
            return SignalDirection.NONE

        sqz_off = df["sqz_off"].iloc[current_idx]
        prev_sqz_off = df["sqz_off"].iloc[current_idx - 1]
        momentum = df["sqz"].iloc[current_idx]

        # If Squeeze Fired (Transition from ON (0) to OFF (1))
        # Note: Depending on pandas_ta version, logic might need to check SQZ_ON transition.
        # Here we check if Squeeze OFF is active (1) and momentum is strong.

        # Simple Logic: If Squeeze is "Off" (Volatility Expanding) and Momentum is strong
        if sqz_off == 1:
            if momentum > 0 and df["sqz"].iloc[current_idx - 1] <= momentum:  # Rising Bullish
                return SignalDirection.BUY
            elif momentum < 0 and df["sqz"].iloc[current_idx - 1] >= momentum:  # Falling Bearish
                return SignalDirection.SELL

        return SignalDirection.NONE

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["sqz", "sqz_on", "sqz_off"]
        sqz = self.__robust_scaler.fit_transform(df[cols])

        return pd.DataFrame(sqz, columns=cols, index=df.index)
