import pandas as pd

from framework.data.data_types import SignalDirection
from framework.data.data_types import TradeSignal
from framework.strategies.abstract_strategy import AbstractStrategy


class MomentumStrategy(AbstractStrategy):
    """
    A basic Trend-Following Strategy using EMA, RSI, and ADX.
    """

    def __init__(self, config: object) -> None:
        self.config = config

    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        """
        Analyzes the latest candle data to generate buy/sell signals.

        Args:
            df: DataFrame with calculated indicators.

        Returns:
            TradeSignal object with decision.
        """
        if df.empty or len(df) < 50:  # Ensure enough data for indicators
            return TradeSignal(SignalDirection.NONE, 0.0, 0.0, 0.0, "Not enough data")

        # Get the latest completed candle
        current = df.iloc[-1]

        # --- Trend Condition ---
        bullish_trend = current["ema_fast"] > current["ema_slow"]
        bearish_trend = current["ema_fast"] < current["ema_slow"]

        # --- Momentum Condition ---
        # Buy only if RSI is not overbought (room to grow)
        rsi_buy_ok = 50 < current["rsi"] < self.config.RSI_OVERBOUGHT
        # Sell only if RSI is not oversold (room to fall)
        rsi_sell_ok = self.config.RSI_OVERSOLD < current["rsi"] < 50

        # --- Volatility Filter (ADX) ---
        # Relaxed for AI Hybrid: We allow ADX > 20 (was 25) to catch earlier trends.
        strong_trend = current["adx"] > 20

        # --- Generate Signal ---
        signal = "none"
        reason = ""

        # BUY LOGIC
        if bullish_trend and rsi_buy_ok and strong_trend:
            signal = "buy"
            reason = f"Trend UP (ADX {current['adx']:.1f}), RSI {current['rsi']:.1f}"

            # Calculate Dynamic Exits using ATR
            stop_loss = current["close"] - (self.config.SL_ATR_MULTIPLIER * current["atr"])
            take_profit = current["close"] + (self.config.TP_ATR_MULTIPLIER * current["atr"])

            return TradeSignal(
                signal,  # type: ignore
                float(current["close"]),
                float(stop_loss),
                float(take_profit),
                reason,
            )

        # SELL LOGIC
        elif bearish_trend and rsi_sell_ok and strong_trend:
            signal = "sell"
            reason = f"Trend DOWN (ADX {current['adx']:.1f}), RSI {current['rsi']:.1f}"

            stop_loss = current["close"] + (self.config.SL_ATR_MULTIPLIER * current["atr"])
            take_profit = current["close"] - (self.config.TP_ATR_MULTIPLIER * current["atr"])

            return TradeSignal(
                signal,  # type: ignore
                float(current["close"]),
                float(stop_loss),
                float(take_profit),
                reason,
            )

        return TradeSignal("none", 0.0, 0.0, 0.0, "No valid signal")
