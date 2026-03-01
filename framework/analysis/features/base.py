from abc import ABC, abstractmethod
from typing_extensions import Self
from typing import Protocol

import numpy as np
import pandas as pd


class Scaler(Protocol):
    def fit(self, X: pd.DataFrame) -> Self: ...
    def transform(self, X: pd.DataFrame) -> np.ndarray: ...
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray: ...


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
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the raw feature data using the fitted scalers.

        Args:
            df (pd.DataFrame): Dataframe with raw feature values.

        Returns:
            pd.DataFrame: Dataframe with columns normalized.
        """
        pass

    def with_scaler(self, scaler: Scaler, **kwargs) -> Self:
        """
        Sets the scaler for the feature.

        Args:
            scaler (Scaler): Scaler to use for the feature.
            **kwargs: Keyword arguments to pass to the scaler.

        Returns:
            Self: The feature with scalers set.
        """
        self._scaler = scaler(**kwargs)
        return self

    def _check_scaler(self) -> None:
        """
        Checks if the scaler is set.

        Raises:
            ValueError: If the scaler is not set.
        """
        if self._scaler is not None:
            return

        raise ValueError("Scaler not set. Use with_scaler() to set the scaler.")
