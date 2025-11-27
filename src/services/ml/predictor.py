import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.keras.models import load_model
from src.api.schemas.predict import PredictionPayload
from src.services.data.preprocessor import DataPreprocessor
from src.config.data_config import DataConfig
from src.services.indicators.technical_indicators import TechnicalIndicatorsService
from src.config.app_config import IndicatorConfig


class ModelPredictor:
    def __init__(self, model_file_path: str):
        self.model = load_model(model_file_path)
        self.data_config = DataConfig()
        self.preprocessor = DataPreprocessor(self.data_config)
        self.indicators_service = TechnicalIndicatorsService(IndicatorConfig())

    def _payload_to_dataframe(self, data: PredictionPayload) -> pd.DataFrame:
        rows = []
        for point in data.sequence:
            rows.append(
                {
                    "Date": pd.to_datetime(point.date),
                    "Open": point.open,
                    "High": point.high,
                    "Low": point.low,
                    "Close": point.close,
                    "Volume": point.volume,
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.indicators_service.add_all_indicators(df)
        return df_with_indicators

    def _prepare_input_sequence(self, df: pd.DataFrame) -> np.ndarray:
        df_processed = self._add_technical_indicators(df)
        scaled_data = self.preprocessor.scale_data(df_processed)

        lookback = self.data_config.lookback_period
        if len(scaled_data) < lookback:
            raise ValueError(
                f"Sequence too short. Need at least {lookback} data points, got {len(scaled_data)}"
            )

        input_sequence = scaled_data[-lookback:]
        input_sequence = input_sequence.reshape(1, lookback, -1)
        return input_sequence

    def predict(self, data: PredictionPayload):
        df = self._payload_to_dataframe(data)
        input_sequence = self._prepare_input_sequence(df)

        predictions = []
        current_sequence = input_sequence.copy()

        for _ in range(data.steps):
            next_prediction = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_prediction[0])

            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_prediction[0]

        predictions_scaled = np.array(predictions)
        predictions_original = self.preprocessor.scaler.inverse_transform(
            predictions_scaled
        )

        last_date = pd.to_datetime(data.sequence[-1].date)
        forecast_dates = [
            (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(data.steps)
        ]

        return {
            "forecast_dates": forecast_dates,
            "predictions": predictions_original.tolist(),
        }
