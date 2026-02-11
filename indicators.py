import config
import pandas as pd
import pandas_ta as ta


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators using pandas_ta and appends them to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data (Open, High, Low, Close, Volume).

    Returns:
        pd.DataFrame: The original dataframe with added indicator columns.
    """
    if df.empty:
        return df

    # --- Trend Indicators ---
    # Fast EMA (9 periods) - Reacts quickly to price changes
    df["ema_fast"] = ta.ema(df["close"], length=config.EMA_FAST_PERIOD)

    # Slow EMA (21 periods) - Shows the broader trend direction
    df["ema_slow"] = ta.ema(df["close"], length=config.EMA_SLOW_PERIOD)

    # ADX (Average Directional Index) - Measures trend STRENGTH (not direction)
    # Returns 3 columns: ADX_14, DMP_14, DMN_14. We only need ADX_14.
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_PERIOD)
    if adx_df is not None:
        df["adx"] = adx_df["ADX_14"]

    # --- Momentum Indicators ---
    # RSI (Relative Strength Index) - Measures overbought/oversold conditions
    df["rsi"] = ta.rsi(df["close"], length=config.RSI_PERIOD)

    # --- Volatility Indicators ---
    # ATR (Average True Range) - Measures market volatility for dynamic Stop Loss
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)

    return df
