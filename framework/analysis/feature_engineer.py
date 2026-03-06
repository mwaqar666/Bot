from typing_extensions import Self
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from framework.analysis.features.base import Feature
from framework.analysis.features.candles import CandleStructure
from framework.analysis.features.momentum import MovingAverageConvergenceDivergence, RelativeStrengthIndex, TTMSqueeze
from framework.analysis.features.overlap import SuperTrend, ExponentialMovingAverage
from framework.analysis.features.performance import DrawDown, LogReturn
from framework.analysis.features.trend import AverageDirectionalIndex, ParabolicStopAndReverse
from framework.analysis.features.volatility import AverageTrueRange, BollingerBands, UlcerIndex
from framework.analysis.features.volume import ChaikinMoneyFlow, MoneyFlowIndex, VolumeRatio


class FeatureEngineer:
    __ohlcv_cols: list[str] = ["open", "high", "low", "close", "volume"]

    def __init__(self) -> None:
        self.__features: list[Feature] = [
            CandleStructure(),
            MovingAverageConvergenceDivergence(),
            RelativeStrengthIndex(),
            TTMSqueeze(),
            SuperTrend(),
            ExponentialMovingAverage(),
            DrawDown(),
            LogReturn(),
            AverageDirectionalIndex(),
            ParabolicStopAndReverse(),
            AverageTrueRange(),
            BollingerBands(),
            UlcerIndex(),
            ChaikinMoneyFlow(),
            MoneyFlowIndex(),
            VolumeRatio(),
        ]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all features to the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        Returns:
            pd.DataFrame: DataFrame with all features added.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(feature.calculate, df) for feature in self.__features]
            results = [f.result() for f in futures]

        results = pd.concat([df[self.__ohlcv_cols], *results], axis=1)
        results.dropna(inplace=True)

        return results

    def fit(self, df: pd.DataFrame) -> Self:
        """
        Fits all scalers in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with feature data.
        Returns:
            Self: The FeatureEngineer instance.
        """

        for feature in self.__features:
            feature.fit(df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all features in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with feature data.
        Returns:
            pd.DataFrame: DataFrame with all features normalized.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(feature.transform, df) for feature in self.__features]
            results = [f.result() for f in futures]

        results = pd.concat([df[self.__ohlcv_cols], *results], axis=1)
        results.dropna(inplace=True)

        return results
