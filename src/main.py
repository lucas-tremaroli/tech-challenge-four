import logging

from data.preparation import load_and_scale_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Application started")
    data_scaled = load_and_scale_data()
    logger.info("Data loaded and scaled successfully")
