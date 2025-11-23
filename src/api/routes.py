from fastapi import APIRouter

from src.utils.service_factory import (
    ModelService,
    ServiceFactory,
    DataLoaderService,
)

router = APIRouter(
    prefix="/api",
    tags=["api"],
)


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/train")
async def train_model(
    data_loader_service: DataLoaderService = ServiceFactory.get_data_loader_service(),
    model_service: ModelService = ServiceFactory.get_model_service(),
):
    """
    Endpoint to train the LSTM model with improved procedures.
    """
    stock_data = data_loader_service.load_data(include_indicators=True)
    scaled_data = data_loader_service.scale_data(stock_data)
    X_train, X_test, y_train, y_test = data_loader_service.get_training_data(
        scaled_data
    )

    model_service = ServiceFactory.get_model_service()

    # Check for data leakage before training
    X_full = data_loader_service.create_sequences(scaled_data)[0]
    y_full = data_loader_service.create_sequences(scaled_data)[1]
    model_service.check_data_leakage(X_full, y_full)

    # Perform time series cross-validation
    _ = model_service.time_series_cross_validation(X_full, y_full)

    # Train final model with improvements
    lstm_model = model_service.build()
    _ = model_service.train(lstm_model, X_train, y_train)

    model_service.evaluate(lstm_model, X_test, y_test)
    lstm_model.save("./assets/models/lstm_model_final.keras")
