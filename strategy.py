from dataclasses import dataclass
import pandas as pd
from typing import Literal


@dataclass
class TradeSignal:
    direction: Literal["buy", "sell", "none"]
    price: float
    stop_loss: float
    take_profit: float
    reason: str


def check_for_signal(df: pd.DataFrame, config) -> TradeSignal:
    """
    Analyzes the latest candle data to generate buy/sell signals.

    Args:
        df: DataFrame with calculated indicators.
        config: Configuration object/module containing parameters.

    Returns:
        TradeSignal object with decision.
    """
    if df.empty or len(df) < 50:  # Ensure enough data for indicators
        return TradeSignal("none", 0, 0, 0, "Not enough data")

    # Get the latest completed candle (row -1 might be current potentially incomplete candle, so use -2 for safety if fetching live but usually -1 is fine if we fetch closed candles)
    # We will assume df contains closed candles.
    current = df.iloc[-1]

    # --- Trend Condition ---
    bullish_trend = current["ema_fast"] > current["ema_slow"]
    bearish_trend = current["ema_fast"] < current["ema_slow"]

    # --- Momentum Condition ---
    # Buy only if RSI is not overbought (room to grow)
    rsi_buy_ok = 50 < current["rsi"] < config.RSI_OVERBOUGHT
    # Sell only if RSI is not oversold (room to fall)
    rsi_sell_ok = config.RSI_OVERSOLD < current["rsi"] < 50

    # --- Volatility Filter (ADX) ---
    # Relaxed for AI Hybrid: We allow ADX > 20 (was 25) to catch earlier trends.
    # The AI will filter bad trades anyway.
    strong_trend = current["adx"] > 20

    # --- Generate Signal ---
    signal = "none"
    reason = ""

    # BUY LOGIC
    # 1. Fast EMA above Slow EMA
    # 2. RSI is healthy (50-70)
    # 3. ADX confirms trend strength > 25
    if bullish_trend and rsi_buy_ok and strong_trend:
        # Check for crossover event (did it just happen?) or continuation
        # For this bot, checking current state is safer than just crossover moment to catch trends
        signal = "buy"
        reason = f"Trend UP (ADX {current['adx']:.1f}), RSI {current['rsi']:.1f}"

        # Calculate Dynamic Exits using ATR
        stop_loss = current["close"] - (config.SL_ATR_MULTIPLIER * current["atr"])
        take_profit = current["close"] + (config.TP_ATR_MULTIPLIER * current["atr"])

        return TradeSignal(signal, current["close"], stop_loss, take_profit, reason)

    # SELL LOGIC
    # 1. Fast EMA below Slow EMA
    # 2. RSI is healthy (30-50)
    # 3. ADX confirms trend strength > 25
    elif bearish_trend and rsi_sell_ok and strong_trend:
        signal = "sell"
        reason = f"Trend DOWN (ADX {current['adx']:.1f}), RSI {current['rsi']:.1f}"

        # Calculate Dynamic Exits using ATR (Shorting: Stop Loss is ABOVE price)
        stop_loss = current["close"] + (config.SL_ATR_MULTIPLIER * current["atr"])
        take_profit = current["close"] - (config.TP_ATR_MULTIPLIER * current["atr"])

        return TradeSignal(signal, current["close"], stop_loss, take_profit, reason)

    return TradeSignal("none", 0, 0, 0, "No valid signal")
