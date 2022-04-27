import hydra
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from keras import layers
from omegaconf import DictConfig
import keras_tuner as kt
from tensorflow import keras
from process import Preprocess


class Model(Preprocess):

    def __innit__(self, config: DictConfig):
        super().__init__(config)

        self.config = config
        self.model = None

    def build_lstm(self):

        lstm = keras.Sequential()
        lstm.add(tf.keras.layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, input_shape=(self.config.process.n_steps_in, self.config.process.n_features))))
        lstm.add(tf.keras.layers.LSTM(units=self.config.model.n_units, activation=self.config.model.activation, dropout=self.config.model.dropout))
        lstm.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        lstm.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=self.config.model.learning_rate))
        self.model = lstm

        return self.model


@hydra.main(config_path="../config", config_name='main')
def build_model(config: DictConfig):
    """Function to build the models"""
    builder = Model(config)
    builder.build_model()

    return builder.model


if __name__ == '__main__':
    build_model()
