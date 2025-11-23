import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from src.services.interfaces.model_service import IModelEvaluator
from src.config.model_config import ModelConfig, PlotConfig


class ModelEvaluator(IModelEvaluator):
    def __init__(self, model_config: ModelConfig, plot_config: PlotConfig):
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        self.plot_config = plot_config
        
        Path(self.plot_config.plots_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray):
        evaluation = model.evaluate(X_test, y_test, verbose=1)

        predictions = model.predict(X_test, verbose=0)

        rmse = self.calculate_rmse(y_test, predictions)
        mape = self.calculate_mape(y_test, predictions)

        self.logger.info(f"Test Loss: {evaluation[0]:.6f}")
        self.logger.info(f"Test MAE: {evaluation[1]:.6f}")
        self.logger.info(f"Test RMSE: {rmse:.6f}")
        self.logger.info(f"Test MAPE: {mape:.2f}%")

        self._plot_predictions(y_test, predictions, rmse, mape)

        return evaluation

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def check_data_leakage(self, X: np.ndarray, y: np.ndarray, lookback: int = 30) -> bool:
        self.logger.info("Checking for data leakage...")

        if len(X) != len(y):
            self.logger.error("X and y have different lengths - potential data preparation issue")
            return False

        sequence_length = X.shape[1] if len(X.shape) > 1 else 1
        if sequence_length != lookback:
            self.logger.warning(f"Sequence length ({sequence_length}) doesn't match lookback ({lookback})")

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.logger.error("Found NaN or infinite values in features")
            return False

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            self.logger.error("Found NaN or infinite values in targets")
            return False

        self.logger.info("Data leakage check passed")
        return True

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mae"], label="Training MAE")
        plt.plot(history.history["val_mae"], label="Validation MAE")
        plt.title("Model MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.plot_config.plots_dir}/{self.plot_config.training_history_filename}")

    def check_overfitting(self, history):
        val_loss = history.history["val_loss"]
        train_loss = history.history["loss"]

        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]

        if final_val_loss > final_train_loss * 1.5:
            self.logger.warning(
                f"Potential overfitting detected: Val Loss ({final_val_loss:.6f}) "
                f">> Train Loss ({final_train_loss:.6f})"
            )

        min_val_loss_epoch = np.argmin(val_loss)
        if min_val_loss_epoch < len(val_loss) - 5:
            self.logger.warning(
                f"Validation loss started increasing at epoch {min_val_loss_epoch}, "
                "consider early stopping"
            )

    def _plot_predictions(self, y_test: np.ndarray, predictions: np.ndarray, rmse: float, mape: float):
        plt.figure(figsize=(15, 10))

        features_to_plot = min(4, y_test.shape[1])

        for i in range(features_to_plot):
            plt.subplot(2, 2, i + 1)

            plt.plot(y_test[:100, i], label="Actual", alpha=0.7)
            plt.plot(predictions[:100, i], label="Predicted", alpha=0.7)

            feature_rmse = self.calculate_rmse(y_test[:, i], predictions[:, i])
            feature_mape = self.calculate_mape(y_test[:, i], predictions[:, i])

            plt.title(f"Feature {i + 1}\nRMSE: {feature_rmse:.4f}, MAPE: {feature_mape:.2f}%")
            plt.xlabel("Time Steps")
            plt.ylabel("Scaled Values")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.plot_config.plots_dir}/{self.plot_config.predictions_filename}", 
                   dpi=self.plot_config.dpi, bbox_inches="tight")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        y_flat = y_test.flatten()
        pred_flat = predictions.flatten()
        plt.scatter(y_flat, pred_flat, alpha=0.5, s=1)
        plt.plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], "r--", lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Overall Predictions\nRMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        residuals = y_flat - pred_flat
        plt.scatter(pred_flat, residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.plot_config.plots_dir}/{self.plot_config.metrics_filename}", 
                   dpi=self.plot_config.dpi, bbox_inches="tight")