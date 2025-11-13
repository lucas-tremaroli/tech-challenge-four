import duckdb
import logging
import numpy as np
import yfinance as yf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_file = "src/data/aapl.db"
        self.db_table = "aapl_stock_data"

    def fetch_stock_data(self, ticker: str, period: str, interval: str):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist

    def load_data(self) -> DataFrame:
        """
        Load stock data from DuckDB database.
        """
        logger.info(f"Loading data from {self.db_file}, table: {self.db_table}")
        with duckdb.connect(self.db_file) as db_con:
            data = db_con.table(self.db_table).to_df()
        return data

    def scale_data(self, data: DataFrame) -> np.ndarray:
        """
        Scale the stock data using Min-Max scaling.
        """
        numeric_columns = data.drop(
            columns=["Date", "Dividends", "Stock Splits"]
        ).columns
        logger.info(f"Scaling columns: {numeric_columns.tolist()}")
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[numeric_columns])
        return data_scaled

    def create_sequences(self, data_scaled: np.ndarray, lookback: int = 60):
        logger.info(f"Creating sequences with lookback period: {lookback}")
        X, y = [], []
        for i in range(len(data_scaled) - lookback):
            X.append(data_scaled[i : i + lookback])
            y.append(data_scaled[i + lookback])
        return np.array(X), np.array(y)

    def get_training_data(self, data_scaled: np.ndarray, lookback: int = 60):
        logger.info("Preparing training data")
        X, y = self.create_sequences(data_scaled, lookback)
        return X, y
