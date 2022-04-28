# ForeML
A package for training and evaluating time series forecasting using Tensorflow.

## :brain: Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make activate
make setup
```

## :mechanical_arm: Install Tensorflow using pip
```bash
pip install tensorflow
```

> To install new PyPI packages, run:
```bash
poetry add <package-name>
```

## :mage: Basic usage
**1. :file_folder: You will find the configurations for your runs in the config folder.**
   Change the paths in the main.yaml file to your personal directories that contain your data. For example:
```bash
raw:
  path: /home/user/ForeML/data/raw/test_data.csv
```
**2. :clipboard: The main.yaml file points to the parameters regarding the data processing and model training:**
```bash
defaults:
  - process: lstm
  - model: lstm
  - forecast: lstm
  - _self_

hypertune: False
```
> If you change the hypertune parameter to True, it will run a tunable model with predefined parameters.

**3. :scissors: The process YAML file listed in the main.yaml should look like this:**
```bash
delimiter: ','
target_index: 5
date_index: 3
n_steps_in: 12
n_steps_out: 8
n_features: 1
```
- First, select the delimiter for your .csv file. After, you specify your target and datetime column indexes.
- Finally, you can input the number of timesteps your model will be trained on (n_steps_in) and the timesteps it will forecast (n_steps_out).
- If your model have more than one feature, you can specify it in the last parameter.

**4. :pushpin: The model YAML file listed in the main.yaml should look like this:**
```bash
name: lstm
type: lstm
n_units: 100
activation: relu
dropout: 0.7
epochs: 10
batch_size: 500
validation_split: 0.2
learning_rate: 0.001
lossfunction: msle
```
> You can change the parameters in this file as you desire. Please note some of those won't cause any effect if you choose to hypertune.

**5. :running_man: When you have all ready and set, you can just run the files you desire:**
```bash
python3 /src/process.py
python3 /src/train_model.py
python3 /src/forecaster.py
```
- process.py will return the .csv with processed data in the right format and the X and y ready to serve as input to the model.
- train_model.py will train the model with your input data and return a trained model as selected (specified architecture and hypertuning option in the config files).
- forecaster.py will take your test data, generate predictions on the dataset and compare with actual values, returning a plot for each timestep.