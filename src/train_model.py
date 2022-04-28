import hydra
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from keras import layers
from omegaconf import DictConfig
import keras_tuner as kt
from tensorflow import keras
from process import Preprocess, LstmModel


class ModelTrainer(Preprocess, LstmModel):
    """Class to train the model"""
    # Initialize the class
    def __init__(self, config: DictConfig):
        # Call the super class constructor
        super().__init__(config)
        self.model = self.get_model()
        self.trained_model = self.train_model()
        self.model = self.hyper_tuning()

        self.config = config
        self.model = None
        self.trained_model = None

    def get_model(self):
        """Function to build the model"""
        if self.config.model.type == 'lstm':
            if self.config.hypertune:
                model = self.build_tunable_lstm()
            else:
                model = self.build_lstm()

        return model

    def checkpoint(self):
        """Function to save the model"""
        checkpoint_filepath = self.config.mlmodel.path
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

        self.trained_model = self.model.fit(self.X, self.y, epochs=self.config.model.epochs, batch_size=self.config.model.batch_size,
                                            validation_split=self.config.model.validation_split, callbacks=[self.checkpoint()])

        return self.trained_model

    def hyper_tuning(self):
        """Function to hyper-tuning the model"""
        if self.config.hypertune:
            tuner = kt.RandomSearch(self.get_model(),
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
            self.model = self.get_model()

            return self.model


@hydra.main(config_path="../config", config_name='main')
def train(config: DictConfig):
    """Script to process the data"""

    print(f"Process data using {config.raw.path}")
    print(f"Parameters used: {config.process.n_steps_in} {config.process.n_steps_out} {config.process.target_index} {config.process.date_index} {config.process.delimiter}")
    trainer = ModelTrainer(config)
    trainer.hyper_tuning()
    trainer.train_model()

    return trainer.trained_model


if __name__ == '__main__':
    train()
