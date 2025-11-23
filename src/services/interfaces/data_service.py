from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd


class IDataFetcher(ABC):
    @abstractmethod
    def fetch_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        pass


class IDataRepository(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame) -> None:
        pass


class IDataPreprocessor(ABC):
    @abstractmethod
    def scale_data(self, data: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def create_sequences(self, data_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def prepare_training_data(self, data_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass