import duckdb
import logging
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_file = "src/data/aapl.db"
        self.db_table = "aapl_stock_data"

    def fetch_stock_data(self, ticker: str, period: str, interval: str):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist

    def load_and_scale_data(self) -> np.ndarray:
        """
        Load stock data from CSV and scale it using MinMaxScaler.
        """
        with duckdb.connect(self.db_file) as db_con:
            data = db_con.table(self.db_table).to_df()
        scaler = MinMaxScaler()
        # Exclude Date column from scaling
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
        data_scaled = scaler.fit_transform(data[numeric_columns])
        return data_scaled

    def create_sequences(self, data_scaled: np.ndarray, lookback: int = 60):
        X, y = [], []
        for i in range(len(data_scaled) - lookback):
            X.append(data_scaled[i : i + lookback])
            y.append(data_scaled[i + lookback])
        return np.array(X), np.array(y)

    def get_training_data(self, lookback: int = 60):
        data_scaled = self.load_and_scale_data()
        X, y = self.create_sequences(data_scaled, lookback)
        return X, y
