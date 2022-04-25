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
        self.data = self.read_data()
        self.X, self.y = self.input_output_split()

    def read_data(self):
        self.data = pd.read_csv(self.config.data_path)
        return self.data

    # split a univariate sequence into samples
    def split_sequence(self):
        X, y = list(), list()
        for i in range(len(self.sequence)):
            # find the end of this pattern
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(self.sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = self.sequence[i:end_ix], self.sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
            self.X = array(X)
            self.y = array(y)
        return self.X, self.y

    def input_output_split(self):
        # define input sequence
        raw_seq = self.data['y']
        # choose a number of time steps
        n_steps_in, n_steps_out = self.n_steps_in, self.n_steps_out,
        # split into samples
        self.X, self.y = self.split_sequence(raw_seq, n_steps_in, n_steps_out)
        # summarize the data
        for i in range(len(self.X)):
            print(self.X[i], self.y[i])


@hydra.main(config_path="../config", config_name='main')
def process_data(config: DictConfig):
    """Function to process the data"""

    raw_path = abspath(config.raw.path)
    print(f"Process data using {raw_path}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == '__main__':
    process_data()
