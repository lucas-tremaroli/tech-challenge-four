from fastapi import APIRouter
from src.api.utils.dependency import container
from src.api.schemas.predict import PredictionPayload, PredictionResponse
from src.monitoring.metrics import (
    track_prediction_metrics,
    track_prediction_steps,
    track_sequence_length
)

router = APIRouter(
    prefix="/api",
)


@router.get("/health", tags=["root"])
async def health_check():
    return {"status": "ok"}


@router.post("/predict", tags=["model"], response_model=PredictionResponse)
@track_prediction_metrics
async def make_prediction(data: PredictionPayload) -> PredictionResponse:
    track_prediction_steps(data.steps)
    track_sequence_length(len(data.sequence))
    
    model_predictor = container.model_predictor()
    result = model_predictor.predict(data)
    return PredictionResponse(**result)
