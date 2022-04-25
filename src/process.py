"""
    This is the demo code that uses hydra to access the parameters in under the directory config.

    Author: Khuyen Tran
"""

import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import pandas as pd
from numpy import array
from datetime import datetime

class preprocess:

    def __init__(self, config):
        self.config = config
    
    def read_data(self):
        data = pd.read_csv(self.config.data_path)
        return data

    def split_data(self, data):
        train_data = data.iloc[:self.config.train_size, :]
        test_data = data.iloc[self.config.train_size:, :]
        return train_data, test_data

    def normalize_data(self, train_data, test_data):
        train_data = train_data.drop(self.config.drop_columns, axis=1)
        test_data = test_data.drop(self.config.drop_columns, axis=1)
        return train_data, test_data

    def get_data(self):
        data = self.read_data()
        train_data, test_data = self.split_data(data)
        train_data, test_data = self.normalize_data(train_data, test_data)
        return train_data, test_data

    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # define input sequence
    raw_seq = df['y']
    # choose a number of time steps
    n_steps_in, n_steps_out = 12, 8
    # split into samples
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])


@hydra.main(config_path="../config", config_name='main')
def process_data(config: DictConfig):
    """Function to process the data"""

    raw_path = abspath(config.raw.path)
    print(f"Process data using {raw_path}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == '__main__':
    process_data()
