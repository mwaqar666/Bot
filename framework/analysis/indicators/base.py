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
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fits the scalers to the indicator data.

        Args:
            df (pd.DataFrame): Dataframe with raw indicator values.

        Returns:
            None
        """
        pass

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the raw indicator data using the fitted scalers for AI/ML input.

        Args:
            df (pd.DataFrame): Dataframe with raw indicator values.

        Returns:
            pd.DataFrame: Dataframe with columns normalized.
        """
        pass
