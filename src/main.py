import logging

from src.services.factory import service_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Application started")
    data_loader = service_factory.get_data_loader_service()
    data = data_loader.load_and_scale_data()
    logger.info("Data loaded and scaled successfully")
