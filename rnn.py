# Recurrent Neural Network

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train=training_set[0:1257]
y_train=training_set[1:1258]

# Reshaping
X_train=np.reshape(X_train,(1257,1,1))

# Building the RNN

# Importing the Keras libraries and packages
    
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM

# Initialising the RNN
regressor=Sequential()

# Adding the input layer and the LSTM layer

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1))) # None for accepting any time step

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,batch_size=32,epochs=200)

# Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values

# Getting the predicted stock price of 2017
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(20,1,1))
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

# Visualising the results

plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Getting the real stock price of 2012-2016

real_stock_price_train=pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train=real_stock_price_train.iloc[:,1:2].values

# Getting the predicted stock price of 2012-2016

predicted_stock_price_train=regressor.predict(X_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)

# Visualising the results

plt.plot(real_stock_price_train,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price_train,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))

# Improving and tuning the RNN

training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)
 
# Creating a data structure with 20 timesteps and t+1 output
X_train=[]
y_train=[]
for i in range(20, 1258):
    X_train.append(training_set[i-20:i, 0])  # The train set needs to be scaled before this
    y_train.append(training_set[i, 0])
X_train, y_train=np.array(X_train), np.array(y_train)

X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 

# Improvised RNN model
model=Sequential()
model.add(LSTM(3,activation='sigmoid',input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
# fit network
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, shuffle=False)

# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
test_set=dataset_test.iloc[:,1:2].values
real_stock_price=np.concatenate((training_set[0:1258], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price=sc.fit_transform(real_stock_price)
inputs=[]
for i in range(1258, 1278):
    inputs.append(scaled_real_stock_price[i-20:i, 0])
inputs=np.array(inputs)
inputs=np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price=model.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[1258:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



