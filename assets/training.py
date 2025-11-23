from src.container import Container
from src.utils.service_factory import ServiceFactory
from src.config.data_config import DataConfig


def main():
    # Initialize dependency injection container
    container = Container()
    container.wire(modules=["src.utils.service_factory"])

    # Get services using the new structure
    data_repository = ServiceFactory.get_data_repository()
    data_preprocessor = ServiceFactory.get_data_preprocessor()
    technical_indicators = ServiceFactory.get_technical_indicators_service()
    model_builder = ServiceFactory.get_model_builder()
    model_trainer = ServiceFactory.get_model_trainer()
    model_evaluator = ServiceFactory.get_model_evaluator()

    # Load configuration
    data_config = DataConfig()

    print("Loading and processing data...")

    # Load raw data
    stock_data = data_repository.load_data()

    # Add technical indicators
    stock_data_with_indicators = technical_indicators.add_all_indicators(stock_data)

    # Scale data
    scaled_data = data_preprocessor.scale_data(stock_data_with_indicators)

    # Prepare training data
    X_train, X_test, y_train, y_test = data_preprocessor.prepare_training_data(
        scaled_data, data_config.lookback_period
    )

    print("Performing data validation...")

    # Check for data leakage before training
    X_full, y_full = data_preprocessor.create_sequences(
        scaled_data, data_config.lookback_period
    )
    model_evaluator.check_data_leakage(X_full, y_full, data_config.lookback_period)

    print("Performing time series cross-validation...")

    # Perform time series cross-validation
    _ = model_trainer.time_series_cross_validation(X_full, y_full)

    print("Training final model...")

    # Build and train final model
    lstm_model = model_builder.build()
    training_history = model_trainer.train(lstm_model, X_train, y_train)

    print("Evaluating model performance...")

    # Plot training history and check for overfitting
    model_evaluator.plot_training_history(training_history)
    model_evaluator.check_overfitting(training_history)

    # Evaluate on test set
    test_results = model_evaluator.evaluate(lstm_model, X_test, y_test)

    print("Saving trained model...")

    # Save the trained model
    lstm_model.save("./assets/models/lstm_model_final.keras")

    print("Training completed successfully!")
    print(f"Final test loss: {test_results[0]:.6f}")
    print(f"Final test MAE: {test_results[1]:.6f}")


if __name__ == "__main__":
    main()
