import logging

from services.factory import ServiceFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Application started")

    data_loader = ServiceFactory.get_data_loader_service()
    stock_data = data_loader.load_data(include_indicators=True)
    scaled_data = data_loader.scale_data(stock_data)
    X, y = data_loader.get_training_data(scaled_data)

    model_service = ServiceFactory.get_model_service()
    lstm_model = model_service.build()
    model_service.train(lstm_model, X, y)
    model_service.evaluate(lstm_model, X, y)

    logger.info("Application finished")
