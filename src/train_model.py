"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from process import Preprocess
import tensorflow as tf
from keras import layers
from tensorflow import keras


class ModelTrainer(Preprocess):

    def __init__(self, config):
        super().__init__(config)

        self.X, self.y = super().yield_data()

    def build_tunable_model(self, hp):

        model = keras.Sequential()
        hp_units = hp.Int('Units Layer 1', min_value=100, max_value=300, step=50)
        dropou = hp.Float('Dropout_rate', min_value=0.5, max_value=0.8, step=0.1)
        lr = hp.Choice('Learning Rate', values=[1e-2, 1e-3])
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(units=hp_units, return_sequences=True, input_shape=self.input_shape)))
        model.add(tf.keras.layers.LSTM(units=hp_units, activation='relu', dropout=dropou))
        model.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        model.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=lr))
        self.model = model

        return self.model

    def build_model(self):

        model = keras.Sequential()
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, input_shape=self.input_shape)))
        model.add(tf.keras.layers.LSTM(units=self.config.process.n_units, activation=self.config.process.activation, dropout=self.config.process.dropout))
        model.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        model.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        self.model = model

        return self.model

    def checkpoint(self):

        checkpoint_filepath = self.config.final_path
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        return model_checkpoint_callback

    def train_model(self):
        """Function to train the model"""
        input_path = abspath(self.config.processed.path)
        output_path = abspath(self.config.final.path)
        model_type = self.config.model.type
        print(f"Train modeling using {input_path}")
        print(f"Model used: {model_type}")
        print(f"Save the output to {output_path}")

        self.trained_model = self.model.fit(self.X, self.y, epochs=self.config.epochs, batch_size=self.config.model.batch_size,
                                            validation_split=self.config.model.validation_split, callbacks=[self.checkpoint()])

        return self.trained_model

    def load_model(self):
        model = keras.models.load_model(self.config.model.name)
        return model

    def hyper_tuning(self):
        if self.config.hypertune:
            tuner = keras.tuner.RandomSearch(self.build_tunable_model(),
                                             objective='val_loss',
                                             max_trials=self.config.model.epochs,
                                             executions_per_trial=1,
                                             directory='/tmp/lstm_tuner',
                                             project_name='lstm_tuner',
                                             overwrite=True,
                                             seed=42)
            tuner.search(self.X, self.Y, epochs=self.config.model.epochs, batch_size=self.config.model.batch_size)
            tuner.results_summary()
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.summary()
            self.model = best_model

            return self.model
        else:
            self.model = self.build_model()

            return self.model


if __name__ == '__main__':
    model = ModelTrainer()
    model = model.hyper_tuning()
    model.train_model()
