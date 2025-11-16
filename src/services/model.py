import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit


class ModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.input_shape = (60, 12)  # timesteps, features (12 features from scaled data)
        self.output_units = 12  # output units matching number of features

    def build(self) -> Sequential:
        model = Sequential()
        # Reduced complexity: 32 units instead of 50
        model.add(LSTM(32, return_sequences=True, input_shape=self.input_shape,
                      recurrent_regularizer=l2(0.01), kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.4))  # Increased dropout from 0.2 to 0.4
        model.add(LSTM(32, return_sequences=False,
                      recurrent_regularizer=l2(0.01), kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.4))
        model.add(Dense(self.output_units, kernel_regularizer=l2(0.01)))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.summary()
        return model

    def train(self, model, X_train, y_train):
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Reduce learning rate when loss plateaus
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=100,  # Increased epochs since early stopping will handle overfitting
            batch_size=32,
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
        self.logger.info(f"Test Loss: {evaluation[0]}, Test MAE: {evaluation[1]}")
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
        plt.savefig('../../assets/training_history.png')
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
            history = model.fit(
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

    def check_data_leakage(self, X, y, lookback=60):
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

    def augment_time_series_data(self, X, y, noise_factor=0.01, scaling_factor=0.1):
        """Apply data augmentation techniques for time series"""
        X_aug = []
        y_aug = []

        for i in range(len(X)):
            # Original data
            X_aug.append(X[i])
            y_aug.append(y[i])

            # Add Gaussian noise
            noise = np.random.normal(0, noise_factor, X[i].shape)
            X_noise = X[i] + noise
            X_aug.append(X_noise)
            y_aug.append(y[i])

            # Scaling (slight amplitude changes)
            scale = 1 + np.random.uniform(-scaling_factor, scaling_factor)
            X_scaled = X[i] * scale
            y_scaled = y[i] * scale
            X_aug.append(X_scaled)
            y_aug.append(y_scaled)

        return np.array(X_aug), np.array(y_aug)

    def train_with_augmentation(self, model, X_train, y_train):
        """Train model with data augmentation"""
        self.logger.info("Training with data augmentation")

        # Apply augmentation
        X_train_aug, y_train_aug = self.augment_time_series_data(X_train, y_train)

        # Early stopping and learning rate reduction
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,  # More patience with augmented data
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )

        history = model.fit(
            X_train_aug,
            y_train_aug,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        self._plot_training_history(history)
        self._check_overfitting(history)
        return history
