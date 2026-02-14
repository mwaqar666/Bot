from abc import ABC, abstractmethod
import pandas as pd

from framework.data.data_types import TradeSignal


class AbstractStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures a consistent interface for the Bot.
    """

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        """
        Analyzes the data and returns a TradeSignal.
        """
        pass
