import hydra
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from keras import layers
from omegaconf import DictConfig
import keras_tuner as kt
from tensorflow import keras
from process import Preprocess


class ModelTrainer(Preprocess):

    def __innit__(self, config: DictConfig):
        super().__init__(config)

        self.config = config
        self.model = None
        self.trained_model = None

    def build_tunable_model(self, hp):

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

    def build_model(self):

        lstm = keras.Sequential()
        lstm.add(tf.keras.layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, input_shape=(self.config.process.n_steps_in, self.config.process.n_features))))
        lstm.add(tf.keras.layers.LSTM(units=self.config.model.n_units, activation=self.config.model.activation, dropout=self.config.model.dropout))
        lstm.add(tf.keras.layers.Dense(self.config.process.n_steps_out))
        lstm.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=self.config.model.learning_rate))
        self.model = lstm

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
        output_path = abspath(self.config.final.path_x)
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
            tuner = kt.RandomSearch(self.build_tunable_model,
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


@hydra.main(config_path="../config", config_name='main')
def train(config: DictConfig):
    """Function to process the data"""

    # instantiate the class
    print(f"Process data using {config.raw.path}")
    print(f"Parameters used: {config.process.n_steps_in} {config.process.n_steps_out} {config.process.target_index} {config.process.date_index} {config.process.delimiter}")
    trainer = ModelTrainer(config)
    trainer.hyper_tuning()
    trainer.train_model()

    return trainer.trained_model


if __name__ == '__main__':
    train()
