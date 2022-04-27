import hydra
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from keras import layers
from omegaconf import DictConfig
import keras_tuner as kt
from tensorflow import keras
from process import Preprocess


class TunableModel(Preprocess):

    def __innit__(self, config: DictConfig):
        super().__init__(config)

        self.config = config
        self.model = None

    def build_tunable_lstm(self, hp):
        lstm = keras.Sequential()
        hp_units = hp.Int('Units Layer 1', min_value=100, max_value=300, step=50)
        dropou = hp.Float('Dropout_rate', min_value=0.5, max_value=0.8, step=0.1)
        lr = hp.Choice('Learning Rate', values=[1e-2, 1e-3])
        lstm.add(tf.keras.layers.Bidirectional(layers.LSTM(units=hp_units, return_sequences=True, input_shape=(self.config.process.n_steps_in, self.config.process.n_features))))
        lstm.add(tf.keras.layers.LSTM(units=hp_units, activation='relu', dropout=dropou))
        lstm.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        lstm.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=lr))
        self.model = lstm

        return self.model


@hydra.main(config_path="../config", config_name='main')
def build_model(config: DictConfig):
    """Function to build the hypertunable models"""
    builder = TunableModel(config)
    builder.build_tunable_lstm()

    return builder.model


if __name__ == '__main__':
    build_model()
