import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from framework.analysis.indicators.base import Indicator
from framework.analysis.indicators.candles import IntraDayCandle
from framework.analysis.indicators.cycles import EvenBetterSineWave
from framework.analysis.indicators.momentum import MovingAverageConvergenceDivergence, RelativeStrengthIndex, TTMSqueeze, PercentageVolumeOscillator, BalanceOfPower, WilliamsR
from framework.analysis.indicators.overlap import SuperTrend, VolumeWeightedAveragePrice
from framework.analysis.indicators.performance import DrawDown, LogReturn
from framework.analysis.indicators.statistics import Entropy
from framework.analysis.indicators.trend import AverageDirectionalIndex, AroonOscillator, ChoppinessIndex, ParabolicStopAndReverse, Vortex
from framework.analysis.indicators.volatility import NormalizedAverageTrueRange, BollingerBands, UlcerIndex
from framework.analysis.indicators.volume import ChaikinMoneyFlow, ElderForceIndex, MoneyFlowIndex, OnBalanceVolume


class TechnicalIndicators:
    def __init__(self) -> None:
        self.__indicators: list[Indicator] = [
            IntraDayCandle(),
            EvenBetterSineWave(),
            MovingAverageConvergenceDivergence(),
            RelativeStrengthIndex(),
            TTMSqueeze(),
            PercentageVolumeOscillator(),
            BalanceOfPower(),
            WilliamsR(),
            SuperTrend(),
            VolumeWeightedAveragePrice(),
            DrawDown(),
            LogReturn(),
            Entropy(),
            AverageDirectionalIndex(),
            AroonOscillator(),
            ChoppinessIndex(),
            ParabolicStopAndReverse(),
            Vortex(),
            NormalizedAverageTrueRange(),
            BollingerBands(),
            UlcerIndex(),
            ChaikinMoneyFlow(),
            ElderForceIndex(),
            MoneyFlowIndex(),
            OnBalanceVolume(),
        ]

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all indicators to the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        Returns:
            pd.DataFrame: DataFrame with all indicators added.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(indicator.calculate, df) for indicator in self.__indicators]
            results = [f.result() for f in futures]

        df = pd.concat([df] + results, axis=1)
        df.dropna(inplace=True)

        return df

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fits all scalers in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with indicator data.
        Returns:
            None
        """

        for indicator in self.__indicators:
            indicator.fit_scaler(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all indicators in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with indicator data.
        Returns:
            pd.DataFrame: DataFrame with all indicators normalized.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(indicator.normalize, df) for indicator in self.__indicators]
            results = [f.result() for f in futures]

        results = pd.concat(results, axis=1).clip(lower=-3, upper=3)
        results.dropna(inplace=True)

        return results
