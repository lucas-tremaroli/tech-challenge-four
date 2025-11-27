from fastapi import APIRouter
from src.api.utils.dependency import container
from src.api.schemas.predict import PredictionPayload, PredictionResponse

router = APIRouter(
    prefix="/api",
)


@router.get("/health", tags=["root"])
async def health_check():
    return {"status": "ok"}


@router.post("/predict", tags=["model"], response_model=PredictionResponse)
async def make_prediction(data: PredictionPayload) -> PredictionResponse:
    model_predictor = container.model_predictor()
    result = model_predictor.predict(data)
    return PredictionResponse(**result)
