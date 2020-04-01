import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dropout
import time #helper libraries
from sklearn.model_selection import train_test_split

input_file = 'Maotai_10.csv'


# convert an array of values into a dataset matrix
def split_X_Y(full_data, look_back=7):
    X_data = np.array([full_data[i:i+look_back, 0] for i in range(len(full_data)-look_back-1)])
    # reshape input to be [samples, time steps, features]
    X_data = X_data.reshape((X_data.shape[0], 1, X_data.shape[1]))
    
    y_data = np.array([full_data[i+look_back, 0] for i in range(len(full_data)-look_back-1)])
    return X_data, y_data


def data_clean(df):
    data = df[~(df == 0).any(axis=1)].dropna()
    return data
    
# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = pd.read_csv(input_file, delimiter=',')
# clean the dataset: drop rows with volume = 0
clean_df = data_clean(df)

start_time = time.time()

# take the adj close price column
y_all = clean_df['Adj Close'].values.reshape(-1, 1)
#dataset=all_y.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler.fit_transform(y_all)

# split into train and test sets, 25% test data, 75% training data
train_data, test_data = train_test_split(y_normalized, test_size=0.2, shuffle=False)

look_back = 7
train_X, train_Y = split_X_Y(train_data, look_back)
test_X, test_Y = split_X_Y(test_data, look_back)

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_X, train_Y, epochs=1000, batch_size=300, verbose=1)

# make predictions
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# invert normalization for the predictions
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(train_Y[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(mean_squared_error(test_Y[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))
error_pct = np.mean(np.abs(test_Y[0] - test_predict[:,0]) / test_Y[0] * 100)
print('Error Percentage: %.2f' % error_pct)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(y_all)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(y_all)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(y_all)-1, :] = test_predict

# plot baseline and predictions
plt.plot(y_all)
plt.plot(trainPredictPlot)

# plot the actual price, prediction in test data=red line, actual price=blue line
plt.plot(testPredictPlot)
plt.show()

end_time = time.time()
print('***** Run time: %s seconds *****' % (end_time - start_time))

# export prediction and actual prices
test_prices= y_all[len(train_predict)+(look_back*2)+1:len(y_all)-1]
df = pd.DataFrame({"date":clean_df['Date'][-len(test_predict)-1:len(y_all)-1],
                   "prediction": list(test_predict.reshape(-1)), 
                   "test_price": list(test_prices.reshape(-1))})
df.to_csv("lstm_result.csv", index=None)
