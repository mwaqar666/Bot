import pandas as pd

# Modular Indicators
from framework.analysis.indicators.momentum import AwesomeOscillator, MACD, RSI, StochasticRSI, TTMSqueeze
from framework.analysis.indicators.overlap import ALMA, EMA, HMA, SuperTrend, VWAP
from framework.analysis.indicators.performance import DrawDown, LogReturn
from framework.analysis.indicators.statistics import Entropy, MAD, StandardDeviation, ZScore
from framework.analysis.indicators.trend import ADX, Aroon, ChoppinessIndex, ParabolicSAR, Vortex
from framework.analysis.indicators.volatility import ATR, BollingerBands, UlcerIndex
from framework.analysis.indicators.volume import CMF, EFI, MFI, OBV, VolumeProfile


class TechnicalIndicators:
    def __init__(self) -> None:
        """
        Initializes the Technical Indicators registry.
        """
        # Momentum Indicators Registry
        self.momentum_indicators = [AwesomeOscillator(), MACD(), RSI(), StochasticRSI(), TTMSqueeze()]

        # Overlap Indicators Registry
        self.overlap_indicators = [ALMA(), EMA(), HMA(), SuperTrend(), VWAP()]

        # Performance Indicators Registry
        self.performance_indicators = [DrawDown(), LogReturn()]

        # Statistics Indicators Registry
        self.statistics_indicators = [Entropy(), MAD(), StandardDeviation(), ZScore()]

        # Trend Indicators Registry
        self.trend_indicators = [ADX(), Aroon(), ChoppinessIndex(), ParabolicSAR(), Vortex()]

        # Volatility Indicators Registry
        self.volatility_indicators = [ATR(), BollingerBands(), UlcerIndex()]

        # Volume Indicators Registry
        self.volume_indicators = [CMF(), EFI(), MFI(), OBV(), VolumeProfile()]

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw OHLCV data into a state vector for the AI.
        Separates indicators into logical groups:
        - Momentum
        - Overlap
        - Performance
        - Statistics
        - Trend
        - Volatility
        - Volume
        """
        df = self.__add_momentum_indicators(df)
        df = self.__add_overlap_indicators(df)
        df = self.__add_performance_indicators(df)
        df = self.__add_statistics_indicators(df)
        df = self.__add_trend_indicators(df)
        df = self.__add_volatility_indicators(df)
        df = self.__add_volume_indicators(df)

        df.dropna(inplace=True)

        return df

    # --- Momentum Indicators ---

    def __add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Momentum Indicators using modular classes."""
        for indicator in self.momentum_indicators:
            df = indicator.calculate(df)

        return df

    # --- Overlap Indicators ---

    def __add_overlap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Overlap Indicators."""
        for indicator in self.overlap_indicators:
            df = indicator.calculate(df)

        return df

    # --- Performance Indicators ---

    def __add_performance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Performance Indicators."""
        for indicator in self.performance_indicators:
            df = indicator.calculate(df)

        return df

    # --- Statistics Indicators ---

    def __add_statistics_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Statistics Indicators."""
        for indicator in self.statistics_indicators:
            df = indicator.calculate(df)

        return df

    # --- Trend Indicators ---

    def __add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Trend Indicators."""
        for indicator in self.trend_indicators:
            df = indicator.calculate(df)

        return df

    # --- Volatility Indicators ---

    def __add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Volatility Indicators."""
        for indicator in self.volatility_indicators:
            df = indicator.calculate(df)

        return df

    # --- Volume Indicators ---

    def __add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Volume Indicators."""
        for indicator in self.volume_indicators:
            df = indicator.calculate(df)

        return df
