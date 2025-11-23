import logging
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.services.interfaces.data_service import IDataPreprocessor
from src.config.data_config import DataConfig


class DataPreprocessor(IDataPreprocessor):
    def __init__(self, config: DataConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.scaler = None

    def scale_data(self, data: DataFrame) -> np.ndarray:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in self.config.exclude_columns]
        
        self.logger.info(f"Scaling columns: {numeric_columns}")
        
        data_clean = data[numeric_columns].dropna()
        
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data_clean)
        else:
            data_scaled = self.scaler.transform(data_clean)
            
        return data_scaled

    def create_sequences(self, data_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info(f"Creating sequences with lookback period: {lookback}")
        X, y = [], []
        for i in range(len(data_scaled) - lookback):
            X.append(data_scaled[i : i + lookback])
            y.append(data_scaled[i + lookback])
        return np.array(X), np.array(y)

    def prepare_training_data(self, data_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self.create_sequences(data_scaled, lookback)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            shuffle=False
        )
        self.logger.info("Data split into training and testing sets")
        return X_train, X_test, y_train, y_test