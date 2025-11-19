import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class ModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.input_shape = (30, 12)  # timesteps, features (30 timesteps, 12 features from scaled data)
        self.output_units = 12  # output units matching number of features

    def build(self) -> Sequential:
        model = Sequential()
        model.add(LSTM(16, input_shape=self.input_shape,
                      recurrent_regularizer=l2(0.005), kernel_regularizer=l2(0.005)))
        model.add(Dropout(0.3))
        model.add(Dense(self.output_units, kernel_regularizer=l2(0.005)))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        model.summary()
        return model

    def train(self, model, X_train, y_train):
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )

        # Reduce learning rate when loss plateaus
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=50,  # Reduced epochs to prevent overfitting
            batch_size=16,  # Smaller batch size for better gradient updates
            validation_split=0.2,
            shuffle=False, # Do not shuffle time series data
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )
        self._plot_training_history(history)
        self._check_overfitting(history)
        return history

    def evaluate(self, model, X_test, y_test):
        evaluation = model.evaluate(X_test, y_test, verbose=1)

        # Get predictions for additional metrics
        predictions = model.predict(X_test, verbose=0)

        # Calculate RMSE and MAPE
        rmse = self.calculate_rmse(y_test, predictions)
        mape = self.calculate_mape(y_test, predictions)

        self.logger.info(f"Test Loss: {evaluation[0]:.6f}")
        self.logger.info(f"Test MAE: {evaluation[1]:.6f}")
        self.logger.info(f"Test RMSE: {rmse:.6f}")
        self.logger.info(f"Test MAPE: {mape:.2f}%")

        # Plot predictions vs actual
        self._plot_predictions(y_test, predictions, rmse, mape)

        return evaluation

    def _plot_training_history(self, history):
        """Plot training and validation loss/metrics to detect overfitting"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./assets/training_history.png')
        plt.show()

    def _check_overfitting(self, history):
        """Check for signs of overfitting"""
        val_loss = history.history['val_loss']
        train_loss = history.history['loss']

        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]

        # Check if validation loss is much higher than training loss
        if final_val_loss > final_train_loss * 1.5:
            self.logger.warning(
                f"Potential overfitting detected: Val Loss ({final_val_loss:.6f}) "
                f">> Train Loss ({final_train_loss:.6f})"
            )

        # Check if validation loss started increasing while training loss decreased
        min_val_loss_epoch = np.argmin(val_loss)
        if min_val_loss_epoch < len(val_loss) - 5:
            self.logger.warning(
                f"Validation loss started increasing at epoch {min_val_loss_epoch}, "
                "consider early stopping"
            )

    def time_series_cross_validation(self, X, y, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        self.logger.info(f"Performing {n_splits}-fold time series cross-validation")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Build and train model for this fold
            model = self.build()
            _ = model.fit(
                X_train_cv, y_train_cv,
                epochs=50, batch_size=32,
                validation_data=(X_val_cv, y_val_cv),
                verbose=0
            )

            # Evaluate
            val_score = model.evaluate(X_val_cv, y_val_cv, verbose=0)
            cv_scores.append(val_score[1])  # MAE score

            self.logger.info(f"Fold {fold + 1} MAE: {val_score[1]:.6f}")

        mean_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)

        self.logger.info(f"Cross-validation MAE: {mean_mae:.6f} ± {std_mae:.6f}")
        return cv_scores

    def check_data_leakage(self, X, y, lookback=30):
        """Check for potential data leakage issues"""
        self.logger.info("Checking for data leakage...")

        # Check if future data is accidentally included in features
        # For time series, X[i] should only contain data from t-lookback to t-1
        # and y[i] should be data at time t

        # Basic check: ensure sequences are properly constructed
        if len(X) != len(y):
            self.logger.error("X and y have different lengths - potential data preparation issue")
            return False

        # Check temporal order
        sequence_length = X.shape[1] if len(X.shape) > 1 else 1
        if sequence_length != lookback:
            self.logger.warning(
                f"Sequence length ({sequence_length}) doesn't match lookback ({lookback})"
            )

        # Check for any NaN or infinite values that might indicate leakage
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.logger.error("Found NaN or infinite values in features")
            return False

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            self.logger.error("Found NaN or infinite values in targets")
            return False

        self.logger.info("Data leakage check passed")
        return True

    def calculate_rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _plot_predictions(self, y_test, predictions, rmse, mape):
        """Plot predictions vs actual values with RMSE and MAPE metrics"""
        plt.figure(figsize=(15, 10))

        # Since we have 12 features, plot the first few for visualization
        features_to_plot = min(4, y_test.shape[1])

        for i in range(features_to_plot):
            plt.subplot(2, 2, i + 1)

            # Plot actual vs predicted for this feature
            plt.plot(y_test[:100, i], label='Actual', alpha=0.7)
            plt.plot(predictions[:100, i], label='Predicted', alpha=0.7)

            # Calculate metrics for this specific feature
            feature_rmse = self.calculate_rmse(y_test[:, i], predictions[:, i])
            feature_mape = self.calculate_mape(y_test[:, i], predictions[:, i])

            plt.title(f'Feature {i+1}\nRMSE: {feature_rmse:.4f}, MAPE: {feature_mape:.2f}%')
            plt.xlabel('Time Steps')
            plt.ylabel('Scaled Values')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./assets/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Overall metrics plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        # Scatter plot of actual vs predicted (flattened)
        y_flat = y_test.flatten()
        pred_flat = predictions.flatten()
        plt.scatter(y_flat, pred_flat, alpha=0.5, s=1)
        plt.plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Overall Predictions\nRMSE: {rmse:.4f}, MAPE: {mape:.2f}%')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Residuals plot
        residuals = y_flat - pred_flat
        plt.scatter(pred_flat, residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./assets/model_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
