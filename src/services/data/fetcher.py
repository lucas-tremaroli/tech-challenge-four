import logging
import yfinance as yf
from pandas import DataFrame
from src.services.interfaces.data_service import IDataFetcher


class YahooFinanceDataFetcher(IDataFetcher):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_stock_data(self, ticker: str, period: str, interval: str) -> DataFrame:
        self.logger.info(
            f"Fetching stock data for {ticker} with period {period} and interval {interval}"
        )
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            self.logger.info(f"Successfully fetched {len(hist)} records for {ticker}")
            return hist
        except Exception as e:
            self.logger.error(f"Failed to fetch stock data for {ticker}: {str(e)}")
            raise
