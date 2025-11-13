import logging
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the dataframe.
        """
        self.logger.info(f"Calculating RSI with period {period}")
        df["rsi"] = ta.rsi(df["Close"], length=period)
        return df

    def add_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicators to the dataframe.
        """
        self.logger.info(f"Calculating MACD with periods {fast}, {slow}, {signal}")
        macd_data = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        df["macd"] = macd_data[f"MACD_{fast}_{slow}_{signal}"]
        df["macd_signal"] = macd_data[f"MACDs_{fast}_{slow}_{signal}"]
        df["macd_histogram"] = macd_data[f"MACDh_{fast}_{slow}_{signal}"]
        return df

    def add_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        """
        self.logger.info(
            f"Calculating Bollinger Bands with period {period} and std {std}"
        )
        bb_data = ta.bbands(df["Close"], length=period, std=std)
        df["bb_lower"] = bb_data[f"BBL_{period}_{std}_{std}"]
        df["bb_middle"] = bb_data[f"BBM_{period}_{std}_{std}"]
        df["bb_upper"] = bb_data[f"BBU_{period}_{std}_{std}"]
        return df

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators (RSI, MACD, Bollinger Bands) to the dataframe.
        """
        self.logger.info("Adding all technical indicators")

        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()

        # Add all indicators
        result_df = self.add_rsi(result_df)
        result_df = self.add_macd(result_df)
        result_df = self.add_bollinger_bands(result_df)

        self.logger.info("All technical indicators added successfully")
        return result_df
