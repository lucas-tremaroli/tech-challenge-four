import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from src.services.interfaces.model_service import IModelBuilder
from src.config.model_config import ModelConfig


class LSTMModelBuilder(IModelBuilder):
    def __init__(self, config: ModelConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def build(self) -> Sequential:
        model = Sequential()
        model.add(
            LSTM(
                self.config.lstm_units,
                input_shape=self.config.input_shape,
                recurrent_regularizer=l2(self.config.l2_regularization),
                kernel_regularizer=l2(self.config.l2_regularization),
            )
        )
        model.add(Dropout(self.config.dropout_rate))
        model.add(
            Dense(
                self.config.output_units,
                kernel_regularizer=l2(self.config.l2_regularization),
            )
        )

        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        model.summary()
        return model

    def get_input_shape(self) -> tuple:
        return self.config.input_shape

    def get_output_units(self) -> int:
        return self.config.output_units
