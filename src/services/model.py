from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class ModelService:
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_units))

        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        model.summary()
        return model

    def train_model(self, model, X_train, y_train, epochs=20, batch_size=32):
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )
        return history
