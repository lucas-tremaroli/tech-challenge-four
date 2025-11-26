import joblib
from src.api.schemas.predict import PredictionPayload


class ModelPredictor:
    def __init__(self, model_file_path: str):
        self.model = joblib.load(model_file_path)

    def predict(self, data: PredictionPayload):
        # Implement prediction logic here
        predictions = self.model.predict(data)
        return predictions
