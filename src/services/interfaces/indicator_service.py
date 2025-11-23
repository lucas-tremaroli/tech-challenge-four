from abc import ABC, abstractmethod
import pandas as pd


class IIndicatorService(ABC):
    @abstractmethod
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass