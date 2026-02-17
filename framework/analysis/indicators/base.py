from abc import ABC, abstractmethod
import pandas as pd
from framework.data.data_types import SignalDirection


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.
    Each indicator must implement its own calculation and signal logic.
    """

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the indicator's columns to the DataFrame.

        Args:
            df (pd.DataFrame): The input OHLCV DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with new indicator columns added.
        """
        pass

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, current_idx: int = -1) -> SignalDirection:
        """
        Analyzes the indicator at the specific index to return a BUY/SELL signal.

        Args:
            df (pd.DataFrame): The DataFrame containing the calculated indicator.
            current_idx (int): The index of the row to analyze (default: last row).

        Returns:
            SignalDirection: BUY, SELL, or NONE.
        """
        pass
