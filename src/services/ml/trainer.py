import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from src.services.interfaces.model_service import IModelTrainer, IModelBuilder
from src.config.model_config import ModelConfig


class ModelTrainer(IModelTrainer):
    def __init__(self, config: ModelConfig, model_builder: IModelBuilder):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_builder = model_builder

    def train(self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray):
        early_stop = EarlyStopping(
            monitor="val_loss", 
            patience=self.config.early_stopping_patience, 
            restore_best_weights=True, 
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", 
            factor=self.config.reduce_lr_factor, 
            patience=self.config.reduce_lr_patience, 
            min_lr=self.config.min_learning_rate, 
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            shuffle=False,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )
        return history

    def time_series_cross_validation(self, X: np.ndarray, y: np.ndarray, n_splits: int = None) -> list:
        if n_splits is None:
            n_splits = self.config.cv_splits
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        self.logger.info(f"Performing {n_splits}-fold time series cross-validation")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model = self.model_builder.build()
            _ = model.fit(
                X_train_cv,
                y_train_cv,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_val_cv, y_val_cv),
                verbose=0,
            )

            val_score = model.evaluate(X_val_cv, y_val_cv, verbose=0)
            cv_scores.append(val_score[1])

            self.logger.info(f"Fold {fold + 1} MAE: {val_score[1]:.6f}")

        mean_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)

        self.logger.info(f"Cross-validation MAE: {mean_mae:.6f} ± {std_mae:.6f}")
        return cv_scores