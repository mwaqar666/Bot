import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime, timedelta

import config

import ccxt
import pandas as pd


class DataLoader:
    """
    Handles fetching and loading data.
    """

    def __init__(self) -> None:
        """
        Initializes the DataLoader with the exchange instance.
        """
        self.__exchange = ccxt.binance({"enableRateLimit": True})

    def fetch_historical_data(self, symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, days: int = config.DATA_LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
        """
        Downloads historical OHLCV data from Binance.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): The timeframe for the candles (e.g., '15m').
            days (int): Number of days of historical data to fetch.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the historical data,
                                    or None if no data was found.
        """
        start_time, end_time = self.__calculate_time_range(days)
        all_candles = self.__fetch_candles(symbol, timeframe, start_time, end_time)

        if not all_candles:
            print("  No data found.")
            return None

        df = self.__convert_to_dataframe(all_candles)

        for col in df.columns:
            df[col] = df[col] if col == "timestamp" else df[col].astype(float)

        return df

    def load_data_from_disk(self, symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, suffix: str = "") -> Optional[pd.DataFrame]:
        """
        Loads historical data from a local CSV file.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): The timeframe for the candles (e.g., '15m').
            suffix (str): Suffix to append to the filename.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame or None if not found.
        """
        filename = self.__get_file_path(symbol, timeframe, suffix)
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None

        print(f"Loading data from disk: {filename}...")
        df = pd.read_csv(filename)

        # Ensure index is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str, suffix: str = "") -> str:
        """
        Saves the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The data to save.
            symbol (str): The trading pair symbol.
            timeframe (str): The timeframe string.
            suffix (str): Suffix to append to the filename.
        """
        filename = self.__get_file_path(symbol, timeframe, suffix)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename)

        print(f"  Success! Saved {len(df)} rows to {filename}")

        return filename

    def __calculate_time_range(self, days: int) -> tuple[datetime, datetime]:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        return start_time, end_time

    def __fetch_candles(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> list[list]:
        """
        Downloads OHLCV candles in parallel chunks for the given time range.

        Args:
            symbol (str): The trading pair symbol.
            timeframe (str): The candle timeframe string.
            start_time (datetime): The inclusive start of the range.
            end_time (datetime): The exclusive end of the range.

        Returns:
            list[list]: A sorted, deduplicated list of OHLCV candles.
        """
        since = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        tf_ms = self.__exchange.parse_timeframe(timeframe) * 1000  # timeframe duration in ms
        chunk_ms = tf_ms * 1000  # ms covered by 1000 candles (max per request)

        chunk_starts = list(range(since, end_timestamp, chunk_ms))
        print(f"Fetching {symbol} ({timeframe}) from {start_time.date()} across {len(chunk_starts)} chunks...")

        raw: list[list] = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            fetch = lambda s: self.__fetch_chunk(symbol, timeframe, s)
            for chunk in executor.map(fetch, chunk_starts):
                raw.extend(chunk)

        unique = {candle[0]: candle for candle in raw}
        return [unique[k] for k in sorted(unique) if k <= end_timestamp]

    def __fetch_chunk(self, symbol: str, timeframe: str, since: int) -> list[list]:
        """
        Fetches a single page of OHLCV candles starting from the given timestamp.

        Args:
            symbol (str): The trading pair symbol.
            timeframe (str): The candle timeframe string.
            since (int): The chunk start timestamp in milliseconds.

        Returns:
            list[list]: A list of OHLCV candles, or an empty list on error.
        """
        try:
            candles = self.__exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if candles:
                dt = datetime.fromtimestamp(candles[-1][0] / 1000)
                print(f"  > Downloaded to: {dt}")
            return candles or []
        except Exception as e:
            print(f"  Error fetching chunk at {since}: {e}")
            return []

    def __convert_to_dataframe(self, candles: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def __get_file_path(self, symbol: str, timeframe: str, suffix: str = "") -> str:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return f"framework/data/{safe_symbol}_{timeframe}{suffix}.csv"
