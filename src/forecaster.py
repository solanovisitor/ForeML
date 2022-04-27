import hydra
import pandas as pd
from tensorflow import keras
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
        """Function to read the data"""
        # read data in .csv formt
        self.data = pd.read_csv(self.config.test.path, delimiter=self.config.process.delimiter, skiprows=1)

        return self.data

    def input_data(self):
        """Function to load the data"""
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
        """Function to split the sequence"""
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
        """Function to split the input and output"""
        # define input sequence
        self.sequence = self.data['y']
        # split into samples
        self.X, self.y = self.split_sequence()
        # summarize the data
        for i in range(len(self.X)):
            print(self.X[i], self.y[i])

        return self.X, self.y

    def yield_data(self):
        """Function to yield the final data"""
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.X, self.y = self.input_output_split()
        n_features = 1
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], n_features))

        return self.X, self.y

    def load_model(self):
        """Function to load the model"""
        model = keras.models.load_model(self.config.mlmodel.path)
        return model

    def predict(self):
        """Function to predict the data"""
        # load model
        model = self.load_model()
        # make predictions
        yhat = model.predict(self.X, verbose=0)
        # summarize the first 5 cases
        for i in range(len(self.X)):
            print(self.X[i], self.y[i], yhat[i])

        return yhat

    def save_prediction(self):
        """Function to save the prediction"""
        # save the prediction
        self.prediction = self.predict()
        # save the prediction
        self.prediction.to_csv(self.config.results.path, index=False)

        return self.prediction

    def save_true_values(self):
        """Function to save the true values"""
        # save the true values
        self.true_values = self.y
        # save the true values
        self.true_values.to_csv(self.config.truevalues.path, index=False)

        return self.true_values

    def plot_prediction(self):
        """Function to plot the prediction"""
        for n in range(7):
            true_values = self.true_values.iloc[:, n]
            pred_values = self.prediction.iloc[:, n]
            df = pd.concat((true_values, pred_values), axis=1, keys=['true', 'pred'])
            df.plot(figsize=(16, 12))


@hydra.main(config_path="../config", config_name='main')
def forecast(config: DictConfig):
    """Function to run the forecast"""
    # create the object
    forecaster = Forecaster(config)
    print("Forecaster Created ...")
    # process the data
    forecaster.input_data()
    forecaster.yield_data()
    forecaster.save_prediction()
    forecaster.save_true_values()
    forecaster.plot_prediction()
    forecaster.save_processed_data()


if __name__ == '__main__':
    forecast()
