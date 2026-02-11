import config
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Dict, Optional


class FeatureEngineer:
    """
    Transforms raw OHLCV data into a state vector for the AI.
    """

    def process_data(
        self, df: pd.DataFrame, additional_dfs: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Main pipeline:
        1. Clean Data
        2. Add Indicators (Base Timeframe)
        3. Add Multi-Timeframe Context (if provided)
        4. Normalize/Scale
        """
        df = df.copy()

        df = self._add_technical_indicators(df)
        df = self._add_price_action_features(df)
        df = self._add_volume_features(df)
        df = self._add_time_features(df)

        # Merge Higher Timeframe Context
        if additional_dfs:
            df = self._merge_higher_timeframes(df, additional_dfs)

        df = self._add_lagged_features(df)
        df = self._normalize_features(df)

        # Drop NaNs created by indicators & merging
        df.dropna(inplace=True)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds basic momentum, trend, and volatility indicators"""
        # Trend
        df["ema_fast"] = ta.ema(df["close"], length=config.EMA_FAST_PERIOD)
        df["ema_slow"] = ta.ema(df["close"], length=config.EMA_SLOW_PERIOD)

        # Momentum
        df["rsi"] = ta.rsi(df["close"], length=config.RSI_PERIOD)
        macd = ta.macd(df["close"])
        # MACD Histogram (Normalized by Price)
        df["macd_hist"] = macd["MACDh_12_26_9"] / df["close"]

        # Volatility
        # ATR (Normalized by Price -> Percent Volatility)
        df["atr"] = (
            ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)
            / df["close"]
        )
        bb = ta.bbands(df["close"], length=config.BBANDS_PERIOD, std=config.BBANDS_STD)
        # BBL (Lower), BBM (Middle), BBU (Upper), BBB (Bandwidth), BBP (Percent)
        # Use built-in Bandwidth calculation (Column 3) to avoid naming issues
        df["bb_width"] = bb.iloc[:, 3] / 100.0

        # Volume (OBV)
        df["obv"] = ta.obv(df["close"], df["volume"])

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Relative Volume (RVOL) to detect breakouts"""
        # Calculate 50-period SMA of Volume
        vol_sma = df["volume"].rolling(window=50).mean()

        # RVOL = Current Volume / Average Volume
        # We add 1 to denominator to avoid division by zero
        df["volume_ratio"] = df["volume"] / (vol_sma + 1)

        return df

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds candlestick shape features"""
        # High-Low Range
        df["candle_range"] = (df["high"] - df["low"]) / df["close"]

        # Body Size
        df["candle_body"] = (df["close"] - df["open"]) / df["open"]

        # Shadows
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[
            "close"
        ]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df[
            "close"
        ]

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes cyclial time features"""
        df["sin_hour"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["sin_day"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["cos_day"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        return df

    def _merge_higher_timeframes(
        self, df: pd.DataFrame, additional_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merges 1h/4h context into 15m dataframe.
        CRITICAL: Uses 'shift(1)' + 'ffill' to avoid lookahead bias.
        """
        print(f"DEBUG: Merging Higher Timeframes... (Base {len(df)} rows)")

        for tf_name, tf_df in additional_dfs.items():
            print(f"  > Processing {tf_name} ({len(tf_df)} rows)...")

            # 1. Calculate indicators on the HTF dataframe first!
            # Context Trend
            ema_fast = ta.ema(tf_df["close"], length=12)
            ema_slow = ta.ema(tf_df["close"], length=26)
            tf_df[f"trend_{tf_name}"] = (ema_fast > ema_slow).astype(int)

            # Context Momentum
            tf_df[f"rsi_{tf_name}"] = ta.rsi(tf_df["close"], length=14) / 100.0

            # Context Volatility
            tf_df[f"atr_{tf_name}"] = ta.atr(
                tf_df["high"], tf_df["low"], tf_df["close"], length=14
            )
            tf_df[f"volatility_{tf_name}"] = tf_df[f"atr_{tf_name}"] / tf_df["close"]

            # 2. Select only the columns we want to merge
            cols_to_merge = [
                f"trend_{tf_name}",
                f"rsi_{tf_name}",
                f"volatility_{tf_name}",
            ]
            subset = tf_df[cols_to_merge].copy()

            # 3. Shift Logic (The "Anti-Cheat" Mechanism)
            # We shift HTF data forward by 1 period.
            # Row 10:00 (which contains 10:00-11:00 data) becomes Row 11:00.
            # So at 11:00, we see the 10:00-11:00 data. Safe.
            subset_shifted = subset.shift(1)

            # 4. Merge
            # Left join on index (timestamp)
            df = df.join(subset_shifted, how="left")

            # 5. Forward Fill
            # Propagate the 1h data to the 15m, 30m, 45m marks
            before_fill = df[cols_to_merge].isna().sum().sum()
            df[cols_to_merge] = df[cols_to_merge].ffill()
            after_fill = df[cols_to_merge].isna().sum().sum()

            print(
                f"    Merged {tf_name}: Filled {before_fill - after_fill} NaNs. Remaining NaNs: {after_fill}"
            )

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds memory features"""
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_4"] = df["log_ret"].rolling(4).sum()
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes raw values to neural-net friendly scales"""
        window = 50

        # Normalize Returns
        df["log_ret_norm"] = (
            df["log_ret"] - df["log_ret"].rolling(window).mean()
        ) / df["log_ret"].rolling(window).std()

        # Normalize RSI (0-100 -> 0-1)
        df["rsi_norm"] = df["rsi"] / 100.0

        # Normalize EMA Distance
        df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

        # Normalize OBV Slope using Tanh to squash outliers (-1 to 1)
        # Raw pct_change can be huge, tanh handles it gracefully.
        df["obv_slope"] = np.tanh(df["obv"].pct_change(periods=5).fillna(0))

        # Normalize Volume Ratio (Log scale to handle spikes)
        # Ratio 1.0 -> 0.0. Ratio 2.0 -> 0.69. Ratio 0.5 -> -0.69.
        df["volume_ratio_norm"] = np.log(df["volume_ratio"] + 0.001)

        return df

    def get_state_columns(self) -> List[str]:
        """
        Returns the list of columns the AI actually sees.
        Now includes: RVOL + HTF Context (Trend, RSI, Volatility per merged timeframe).
        This is dynamic based on merged columns availability if needed, but we hardcode expectation.
        """
        base_features = [
            "log_ret_norm",
            "rsi_norm",
            "ema_spread",
            "macd_hist",
            "obv_slope",
            "volume_ratio_norm",  # Volume
            "atr",
            "bb_width",
            "candle_range",
            "candle_body",
            "upper_shadow",
            "lower_shadow",
            "sin_hour",
            "cos_hour",
            "sin_day",
            "cos_day",
            "log_ret_4",
        ]

        # We assume 30m, 1h and 4h are merged for "production" model
        htf_features = [
            "trend_30m",
            "rsi_30m",
            "volatility_30m",
            "trend_1h",
            "rsi_1h",
            "volatility_1h",
            "trend_4h",
            "rsi_4h",
            "volatility_4h",
        ]

        return base_features + htf_features
