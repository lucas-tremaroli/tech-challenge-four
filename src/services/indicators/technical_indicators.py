import logging
import pandas as pd
import pandas_ta as ta
from src.services.interfaces.indicator_service import IIndicatorService
from src.config.app_config import IndicatorConfig


class TechnicalIndicatorsService(IIndicatorService):
    def __init__(self, config: IndicatorConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def add_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        period = period or self.config.rsi_period
        self.logger.info(f"Calculating RSI with period {period}")
        result_df = df.copy()
        result_df["rsi"] = ta.rsi(df["Close"], length=period)
        return result_df

    def add_macd(self, df: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal
        
        self.logger.info(f"Calculating MACD with periods {fast}, {slow}, {signal}")
        result_df = df.copy()
        macd_data = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        result_df["macd"] = macd_data[f"MACD_{fast}_{slow}_{signal}"]
        result_df["macd_signal"] = macd_data[f"MACDs_{fast}_{slow}_{signal}"]
        result_df["macd_histogram"] = macd_data[f"MACDh_{fast}_{slow}_{signal}"]
        return result_df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = None, std: float = None) -> pd.DataFrame:
        period = period or self.config.bollinger_period
        std = std or self.config.bollinger_std
        
        self.logger.info(f"Calculating Bollinger Bands with period {period} and std {std}")
        result_df = df.copy()
        bb_data = ta.bbands(df["Close"], length=period, std=std)
        result_df["bb_lower"] = bb_data[f"BBL_{period}_{std}_{std}"]
        result_df["bb_middle"] = bb_data[f"BBM_{period}_{std}_{std}"]
        result_df["bb_upper"] = bb_data[f"BBU_{period}_{std}_{std}"]
        return result_df

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Adding all technical indicators")
        
        result_df = df.copy()
        result_df = self.add_rsi(result_df)
        result_df = self.add_macd(result_df)
        result_df = self.add_bollinger_bands(result_df)
        
        self.logger.info("All technical indicators added successfully")
        return result_df