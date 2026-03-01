import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from framework.analysis.features.base import Feature


class FeatureEngineer:
    def __init__(self) -> None:
        self.__features: list[Feature] = []

    def add(self, df: pd.DataFrame, features: list[Feature] = None) -> pd.DataFrame:
        """
        Adds all features to the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            features (list[Feature]): List of features to add.
        Returns:
            pd.DataFrame: DataFrame with all features added.
        """

        if features is None:
            return df

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(feature.calculate, df) for feature in features]
            results = [f.result() for f in futures]

        self.__features.extend(features)

        results = pd.concat([df, *results], axis=1)
        results.dropna(inplace=True)

        return results

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits all scalers in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with feature data.
        Returns:
            None
        """

        for feature in self.__features:
            feature.fit(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all features in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with feature data.
        Returns:
            pd.DataFrame: DataFrame with all features normalized.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(feature.normalize, df) for feature in self.__features]
            results = [f.result() for f in futures]

        results = pd.concat([df, *results], axis=1)
        results.dropna(inplace=True)

        return results
