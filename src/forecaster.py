import hydra
import pandas as pd
from numpy import array
from omegaconf import DictConfig
from process import Preprocess


class Forecaster(Preprocess):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.config = config
        self.data = None
        self.sequence = None
        self.X = None
        self.y = None


    def read_data(self):
        # read data in .csv formt
        self.data = pd.read_csv(self.config.raw.path, delimiter=self.config.process.delimiter, skiprows=1)

        return self.data

    def input_data(self):
        # process data into readeble format
        df = self.read_data()
        df['y'] = df.iloc[:, self.config.process.target_index]
        df['ds'] = df.iloc[:, self.config.process.date_index]
        rest = df.drop(['y', 'ds'], axis=1)
        df.drop(rest, inplace=True, axis=1)
        df['ds'].drop_duplicates(inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df.sort_values(by=['ds'], inplace=True)
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.data = df

        return self.data

    def split_sequence(self):

        X, y = list(), list()
        for i in range(len(self.sequence)):
            # find the end of this pattern
            end_ix = i + self.config.process.n_steps_in
            out_end_ix = end_ix + self.config.process.n_steps_out
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
        self.sequence = self.data['y']
        # split into samples
        self.X, self.y = self.split_sequence()
        # summarize the data
        for i in range(len(self.X)):
            print(self.X[i], self.y[i])

        return self.X, self.y

    def yield_data(self):
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.X, self.y = self.input_output_split()
        n_features = 1
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], n_features))

        return self.X, self.y

    def save_processed_data(self):
        # save processed data
        self.data.to_csv(self.config.processed.path, index=False)

    def save_final_data(self):
        # save final data
        X_data = pd.DataFrame(self.X.reshape(self.X.shape[0], self.X.shape[1]))
        y_data = pd.DataFrame(self.y)
        X_data.to_csv(f'{self.config.final.path_x}', index=False)
        y_data.to_csv(self.config.final.path_y, index=False)


@hydra.main(config_path="../config", config_name='main')
def forecast(config: DictConfig):
    """Function to process the data"""

    # instantiate the class
    print(f"Process data using {config.raw.path}")
    print(f"Parameters used: {config.process.n_steps_in} {config.process.n_steps_out} {config.process.target_index} {config.process.date_index} {config.process.delimiter}")
    forecaster = Forecaster(config)
    print("Created ...")
    forecaster.save_processed_data()
    print(f"Processed data saved on {config.processed.path}")
    forecaster.save_final_data()
    print(f"X and y saved on {config.final.path_x} and {config.final.path_y}")


if __name__ == '__main__':
    forecast()
