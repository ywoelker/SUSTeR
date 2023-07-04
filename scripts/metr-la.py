import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from argparse import ArgumentParser

"""
This file generates a sparse dataset from the METR-LA data.
The dropout rate is variable an can be changed.
For the experiments with SUSTeR we use the common 12 steps lead time and predicted one single future step.
Also we divided the data into 70% train, 10% validation and 20% test sets.
"""


ap = ArgumentParser()
ap.add_argument('--dropout', help= 'The dropout for the newly created dataset.', default=0.99, type=float)
args = ap.parse_args()

dropout_ratio = args.dropout


FUTURE_STEPS = 1
LEAD_STEPS = 12
TRAIN = .7
VALIDATION = .1
DATA_FILE_PATH = Path('data/raw/METR-LA/')
OUTPUT_PATH = Path(f'data/METR-LA/DROP{dropout_ratio:0.2f}')




## create an empty folder for the new dataset.
OUTPUT_PATH.parent.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir()


## read data
df = pd.read_hdf(DATA_FILE_PATH / 'METR-LA.h5')
data = np.expand_dims(df.values, axis=-1)
data = data[..., [0]]
print("raw time series shape: {0}".format(data.shape))


l, n, f = data.shape
num_samples = l - (LEAD_STEPS + FUTURE_STEPS) + 1
train_num_short = round(num_samples * TRAIN)
valid_num_short = round(num_samples * VALIDATION)
test_num_short = num_samples - train_num_short - valid_num_short
print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))

index_list = []
for t in range(LEAD_STEPS, num_samples + LEAD_STEPS):
    index = (t-LEAD_STEPS, t, t+FUTURE_STEPS)
    index_list.append(index)

train_index = index_list[:train_num_short]
valid_index = index_list[train_num_short: train_num_short + valid_num_short]
test_index = index_list[train_num_short +
                        valid_num_short: train_num_short + valid_num_short + test_num_short]

scaler = StandardScaler().fit(data[:train_index[-1][1], ...].reshape((train_index[-1][1], -1)))

with open(OUTPUT_PATH / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

data_norm = scaler.transform(data.reshape((data.shape[0], -1))).reshape(data.shape)


## create the feature list
feature_list = [data_norm]

## numerical time_of_day
tod = (
    df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
feature_list.append(tod_tiled)

## numerical day_of_week
dow = df.index.dayofweek / 7 # Scaling to 0 to 1
dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
feature_list.append(dow_tiled)

## read the sensor locations and norm the position
sensor_locations = pd.read_csv(DATA_FILE_PATH / 'graph_sensor_locations.csv')
sensor_locations_for_data = sensor_locations[['latitude', 'longitude']].values
position_scaler = MinMaxScaler((-1, 1)).fit(sensor_locations_for_data[:train_index[-1][1]])
sensor_locations_for_data = position_scaler.transform(sensor_locations_for_data)

sensor_locations_for_data = np.expand_dims(sensor_locations_for_data, axis = 0)
sensor_locations_for_data = np.tile(sensor_locations_for_data, (l, 1, 1))

feature_list.append(sensor_locations_for_data)


## fuse the all the different data products together
processed_data = np.concatenate(feature_list, axis=-1) # l x n x f


mask = np.zeros_like(processed_data)
assert .0 <= dropout_ratio <= 1.0
for i in range(l):
    indices = np.argwhere(np.random.rand(n) > dropout_ratio).squeeze()
    mask[i, indices] = 1

processed_drop_data = np.multiply(processed_data, mask)


## save the data to an index file and a data file
index = {}
index["train"] = train_index
index["valid"] = valid_index
index["test"] = test_index
with open( OUTPUT_PATH / 'index_in{0}_out{1}.pkl'.format(LEAD_STEPS, FUTURE_STEPS), "wb") as f:
    pickle.dump(index, f)

data = {}
data["processed_data"] = processed_data
data["processed_drop_data"] = processed_drop_data

data["scaler_positions"] = (position_scaler.data_min_, position_scaler.data_max_)
data["scaler_values"] = (scaler.mean_, scaler.scale_)


with open(OUTPUT_PATH / f'data_in{LEAD_STEPS}_out{FUTURE_STEPS}.pkl', "wb") as f:
    pickle.dump(data, f)