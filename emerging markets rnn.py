# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('ML-EMHYY.csv')
dataset_train.sort_values(['DATE'], ascending = True, inplace= True)

#---EDA---
dataset_train.describe()
dataset_train.info()

dataset_train['DATE'] = pd.to_datetime(dataset_train['DATE'])
dataset_train.set_index('DATE',inplace = True)
plt.plot(dataset_train)

training_set = dataset_train.values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 4000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.3))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Creating a data structure with 60 timesteps and 1 output
X_test = []
y_test = []    

for i in range(3940, 5468):
    X_test.append(training_set_scaled[i-60:i, 0])
    y_test.append(training_set_scaled[i, 0])
X_test, y_test= np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

pred = regressor.predict(X_test)

y_test = y_test.reshape(1528,1)
y_test = sc.inverse_transform(y_test)
pred = sc.inverse_transform(pred)

# Visualising the results
plt.plot(y_test, color = 'red', label = 'Real')
plt.plot(pred, color = 'blue', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Yield')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, pred))

from keras.losses import mean_absolute_percentage_error
mean_absolute_percentage_error = mean_absolute_percentage_error(y_test, pred)

