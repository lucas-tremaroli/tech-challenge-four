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
    X_train, X_test, y_train, y_test = data_loader.get_training_data(scaled_data)

    model_service = ServiceFactory.get_model_service()
    
    # Check for data leakage before training
    X_full = data_loader.create_sequences(scaled_data)[0]
    y_full = data_loader.create_sequences(scaled_data)[1]
    model_service.check_data_leakage(X_full, y_full)
    
    # Perform time series cross-validation
    cv_scores = model_service.time_series_cross_validation(X_full, y_full)
    
    # Train final model with improvements
    lstm_model = model_service.build()
    
    # Option 1: Regular training with regularization and early stopping
    logger.info("Training with regularization and early stopping...")
    history = model_service.train(lstm_model, X_train, y_train)
    
    # Option 2: Train with data augmentation (uncomment to use)
    # logger.info("Training with data augmentation...")
    # history = model_service.train_with_augmentation(lstm_model, X_train, y_train)
    
    model_service.evaluate(lstm_model, X_test, y_test)

    logger.info("Application finished")
