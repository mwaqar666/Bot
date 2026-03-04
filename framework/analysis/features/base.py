from abc import ABC, abstractmethod
from typing_extensions import Self

import pandas as pd


class Feature(ABC):
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the feature columns to the DataFrame.

        Args:
            df (pd.DataFrame): The input OHLCV DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with new feature columns added.
        """
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        """
        Fits the feature to the data, and preserves the scaler state.

        Args:
            df (pd.DataFrame): Dataframe with raw feature values.

        Returns:
            None
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the raw feature data using the fitted scalers.

        Args:
            df (pd.DataFrame): Dataframe with raw feature values.

        Returns:
            pd.DataFrame: Dataframe with columns transformed.
        """
        pass
