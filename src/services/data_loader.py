import duckdb
import logging
import numpy as np
import yfinance as yf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataLoaderService:
    def __init__(self, technical_indicators=None):
        self.logger = logging.getLogger(__name__)
        self.db_file = "src/data/aapl.db"
        self.db_table = "aapl_stock_data"
        self.technical_indicators = technical_indicators

    def fetch_stock_data(self, ticker: str, period: str, interval: str):
        """
        Fetch stock data from Yahoo Finance.
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist

    def load_data(self, include_indicators: bool = False) -> DataFrame:
        """
        Load stock data from DuckDB database.
        """
        self.logger.info(f"Loading data from {self.db_file}, table: {self.db_table}")
        with duckdb.connect(self.db_file) as db_con:
            data = db_con.table(self.db_table).to_df()

        if include_indicators and self.technical_indicators:
            self.logger.info("Adding technical indicators to the data")
            data = self.technical_indicators.add_all_indicators(data)

        return data

    def scale_data(self, data: DataFrame) -> np.ndarray:
        """
        Scale the stock data using Min-Max scaling.
        """
        # Identify columns to exclude from scaling
        exclude_columns = ["Date", "Dividends", "Stock Splits"]

        # Get numeric columns (exclude text/date columns and handle NaN from indicators)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

        self.logger.info(f"Scaling columns: {numeric_columns}")

        # Drop rows with NaN values (from technical indicators) before scaling
        data_clean = data[numeric_columns].dropna()

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_clean)
        return data_scaled

    def create_sequences(self, data_scaled: np.ndarray, lookback: int = 30):
        """
        Create sequences of data for time series forecasting.
        """
        self.logger.info(f"Creating sequences with lookback period: {lookback}")
        X, y = [], []
        for i in range(len(data_scaled) - lookback):
            X.append(data_scaled[i : i + lookback])
            y.append(data_scaled[i + lookback])
        return np.array(X), np.array(y)

    def get_training_data(self, data_scaled: np.ndarray, lookback: int = 30):
        """
        Prepare training data for the model.
        """
        X, y = self.create_sequences(data_scaled, lookback)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.logger.info("Data split into training and testing sets")
        return X_train, X_test, y_train, y_test
