import logging
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_data_path = "src/data/processed/training_data.csv"

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
