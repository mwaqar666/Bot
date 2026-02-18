import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from framework.analysis.indicators.base import Indicator
from framework.analysis.indicators.momentum import AwesomeOscillator, MovingAverageConvergenceDivergence, RelativeStrengthIndex, StochasticRSI, TTMSqueeze
from framework.analysis.indicators.overlap import ArnaudLegouxMovingAverage, ExponentialMovingAverage, HullMovingAverage, SuperTrend, VolumeWeightedAveragePrice
from framework.analysis.indicators.performance import DrawDown, LogReturn
from framework.analysis.indicators.statistics import Entropy, MeanAbsoluteDeviation, StandardDeviation, ZScore
from framework.analysis.indicators.trend import AverageDirectionalIndex, AroonOscillator, ChoppinessIndex, ParabolicStopAndReverse, Vortex
from framework.analysis.indicators.volatility import AverageTrueRange, BollingerBands, UlcerIndex
from framework.analysis.indicators.volume import ChaikinMoneyFlow, ElderForceIndex, MoneyFlowIndex, OnBalanceVolume


class TechnicalIndicators:
    def __init__(self) -> None:
        """
        Initializes the Technical Indicators registry.
            - Momentum
            - Overlap
            - Performance
            - Statistics
            - Trend
            - Volatility
            - Volume
        """
        self.__indicators: list[Indicator] = [
            AwesomeOscillator(),
            MovingAverageConvergenceDivergence(),
            RelativeStrengthIndex(),
            StochasticRSI(),
            TTMSqueeze(),
            ArnaudLegouxMovingAverage(),
            ExponentialMovingAverage(),
            HullMovingAverage(),
            SuperTrend(),
            VolumeWeightedAveragePrice(),
            DrawDown(),
            LogReturn(),
            Entropy(),
            MeanAbsoluteDeviation(),
            StandardDeviation(),
            ZScore(),
            AverageDirectionalIndex(),
            AroonOscillator(),
            ChoppinessIndex(),
            ParabolicStopAndReverse(),
            Vortex(),
            AverageTrueRange(),
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

        ohlcv = df[["open", "high", "low", "close", "volume"]]
        df = pd.concat([ohlcv] + results, axis=1)
        df.dropna(inplace=True)

        return df

    def normalize_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all indicators in the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        Returns:
            pd.DataFrame: DataFrame with all indicators normalized.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(indicator.normalize, df) for indicator in self.__indicators]
            results = [f.result() for f in futures]

        ohlcv = df[["open", "high", "low", "close", "volume"]]
        df = pd.concat([ohlcv] + results, axis=1).clip(lower=-3, upper=3)
        df.dropna(inplace=True)

        return df
