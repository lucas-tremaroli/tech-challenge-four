from fastapi import APIRouter
from src.api.utils.dependency import container
from src.api.schemas.predict import PredictionPayload

router = APIRouter(
    prefix="/api",
)


@router.get("/health", tags=["root"])
async def health_check():
    return {"status": "ok"}


@router.post("/predict", tags=["model"])
async def make_prediction(data: PredictionPayload):
    model_predictor = container.model_predictor()
    predictions = model_predictor.predict(data)
    return {"predictions": predictions}
