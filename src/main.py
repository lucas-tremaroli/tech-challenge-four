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
    scaled_data = data_loader.load_and_scale_data()
    logger.info(scaled_data.head())

    logger.info("Application finished")
