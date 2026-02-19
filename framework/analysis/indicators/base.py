from abc import ABC, abstractmethod
import pandas as pd


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
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the indicator columns for AI/ML input.
        Default implementation returns the DataFrame unchanged.
        Override this in subclasses to apply scaling (e.g. Z-Score, MinMax).

        Args:
            df (pd.DataFrame): Dataframe with raw indicator values.

        Returns:
            pd.DataFrame: Dataframe with normalized columns (e.g. 'mfi_norm').
        """
        pass
