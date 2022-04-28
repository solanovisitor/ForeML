import hydra
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from keras import layers
from omegaconf import DictConfig
import keras_tuner as kt
from tensorflow import keras
from process import Preprocess


class Model(Preprocess):
    """Class to define the models architectures"""
    def __innit__(self, config: DictConfig):
        super().__init__(config)

        self.config = config
        self.model = None

    def build_lstm(self):
        """Function to build the LSTM model"""
        lstm = keras.Sequential()
        lstm.add(tf.keras.layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, input_shape=(self.config.process.n_steps_in, self.config.process.n_features))))
        lstm.add(tf.keras.layers.LSTM(units=self.config.model.n_units, activation=self.config.model.activation, dropout=self.config.model.dropout))
        lstm.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        lstm.compile(loss=self.config.model.lossfunction, optimizer=keras.optimizers.Adam(learning_rate=self.config.model.learning_rate))
        self.model = lstm

        return self.model

    def build_tunable_model(self, hp):

        model = keras.Sequential()
        hp_units = hp.Int('Units Layer 1', min_value=200, max_value=300, step=50)
        dropou = hp.Float('Dropout_rate', min_value=0.5, max_value=0.7, step=0.1)
        lr = hp.Choice('Learning Rate', values=[1e-2, 1e-3])
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(units=hp_units, return_sequences=True, input_shape=(self.config.process.n_steps_in, self.config.process.n_features))))
        model.add(tf.keras.layers.LSTM(units=hp_units, activation=self.config.model.activation, dropout=dropou))
        model.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        model.compile(loss=self.config.model.lossfunction, optimizer=keras.optimizers.Adam(learning_rate=lr))
        return model


@hydra.main(config_path="../config", config_name='main')
def build_model(config: DictConfig):
    """Function to build the models"""
    builder = Model(config)
    builder.build_model()

    return builder.model


if __name__ == '__main__':
    build_model()
