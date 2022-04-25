import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from keras import layers


class lstm:

    def __init__(self, hypertune, n_of_layers, learning_rate, dropout, date_index, target_index, is_multivariate, input_shape) -> None:

        self.hypertune = hypertune
        self.n_of_layers = n_of_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.date_index = date_index
        self.target_index = target_index
        self.is_multivariate = is_multivariate
        self.input_shape = input_shape

    def build_tunable_model(self, hp):

        model = keras.Sequential()
        hp_units = hp.Int('Units Layer 1', min_value=100, max_value=300, step=50)
        dropou = hp.Float('Dropout_rate', min_value=0.5, max_value=0.8, step=0.1)
        lr = hp.Choice('Learning Rate', values=[1e-2, 1e-3])
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(units=hp_units, return_sequences=True, input_shape=self.input_shape)))
        model.add(tf.keras.layers.LSTM(units=hp_units, activation='relu', dropout=dropou))
        model.add(tf.keras.layers.Dense(8))
        model.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=lr))

        return model

    def build_model(self):
        model = keras.Sequential()
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, input_shape=self.input_shape)))
        model.add(tf.keras.layers.LSTM(units=100, activation='relu', dropout=0.5))
        model.add(tf.keras.layers.Dense(8))
        model.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=0.001))

        return model

    def train_model(self, model, train_data, test_data, epochs, batch_size):

        model.fit(train_data, epochs=epochs, batch_size=batch_size)
        test_loss = model.evaluate(test_data)

        return test_loss

    def predict_model(self, model, test_data):
        return model.predict(test_data)

    def save_model(self, model, model_name):
        model.save(model_name)

    def load_model(self, model_name):
        model = keras.models.load_model(model_name)
        return model

    def hyper_tuning(self, train_data, test_data, epochs, batch_size):
        if self.hypertune:
            tuner = keras.tuner.RandomSearch(self.build_tunable_model,
                                             objective='val_loss',
                                             max_trials=10,
                                             executions_per_trial=1,
                                             directory='/tmp/lstm_tuner',
                                             project_name='lstm_tuner',
                                             overwrite=True,
                                             seed=42)
            tuner.search(train_data, test_data, epochs=epochs, batch_size=batch_size)
            tuner.results_summary()
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.summary()
            return best_model
        else:
            return self.build_model(None)

    def run(self, train_data, test_data, epochs, batch_size):
        model = self.load_model('models/lstm_model.h5')
        if self.hypertune:
            model = self.hyper_tuning(train_data, test_data, epochs, batch_size)
        test_loss = self.train_model(model, train_data, test_data, epochs, batch_size)
        print(test_loss)
        self.save_model(model, 'models/lstm_model.h5')
        return test_loss

    def run_predict(self, test_data):
        model = self.load_model('models/lstm_model.h5')
        test_pred = self.predict_model(model, test_data)
        return test_pred

    def run_predict_multivariate(self, test_data):
        model = self.load_model('models/lstm_model.h5')
        test_pred = self.predict_model(model, test_data)
        return test_pred
