import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt    
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Data Extraction
mt5.initialize(
    path = "C:\\Users\\silin\\AppData\\Roaming\\MetaTrader 5\\terminal64.exe",
    login = 0000000,
    password = "*******"
    )

# query data
timezone = pytz.timezone("Etc/UTC")
utc_from = datetime(2019, 12, 31, tzinfo=timezone)

rates = mt5.copy_rates_from('EURUSD', mt5.TIMEFRAME_H1, utc_from, 50000)

#off connection
mt5.shutdown()

# cleanup data
eurusd = pd.DataFrame(rates)
eurusd['time']=pd.to_datetime(eurusd['time'], unit='s')


# fix random seed for reproducibility
np.random.seed(7)

dataset = eurusd.loc[:,['close']].values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(64, input_shape=(1, look_back)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=50, verbose=2)


model.save('LSTM_eurusd.model')



