from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.models import Sequential


class IModelBuilder(ABC):
    @abstractmethod
    def build(self) -> Sequential:
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        pass

    @abstractmethod
    def get_output_units(self) -> int:
        pass


class IModelTrainer(ABC):
    @abstractmethod
    def train(self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def time_series_cross_validation(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> list:
        pass


class IModelEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray):
        pass

    @abstractmethod
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def check_data_leakage(
        self, X: np.ndarray, y: np.ndarray, lookback: int = 30
    ) -> bool:
        pass
