import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_data_path = "src/data/aapl_stock_data.csv"

    def fetch_stock_data(self, ticker: str, period: str, interval: str):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist

    def load_and_scale_data(self):
        data = pd.read_csv(self.training_data_path)
        scaler = MinMaxScaler()
        # Exclude Date column from scaling
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
        data_scaled = scaler.fit_transform(data[numeric_columns])
        return data_scaled

    def create_sequences(self, data_scaled: np.ndarray, seq_length: int = 60):
        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i : i + seq_length])
            y.append(data_scaled[i + seq_length])
        return np.array(X), np.array(y)

    def get_training_data(self, seq_length: int = 60):
        data_scaled = self.load_and_scale_data()
        X, y = self.create_sequences(data_scaled, seq_length)
        return X, y
