import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class ModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.input_shape = (60, 12)  # timesteps, features (12 features from scaled data)
        self.output_units = 12  # output units matching number of features

    def build(self) -> Sequential:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_units))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.summary()
        return model

    def train(self, model, X_train, y_train):
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )
        return history

    def evaluate(self, model, X_test, y_test):
        evaluation = model.evaluate(X_test, y_test, verbose=1)
        self.logger.info(f"Test Loss: {evaluation[0]}, Test MAE: {evaluation[1]}")
        return evaluation
