import logging
from sklearn.model_selection import train_test_split

from services.factory import service_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Application started")
    data_loader = service_factory.get_data_loader_service()
    training_data = data_loader.get_training_data()

    # Split data into features and labels
    X_train, X_temp, y_train, y_temp = train_test_split(
        training_data[0], training_data[1], test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logger.info("Data split into training, validation, and test sets")
