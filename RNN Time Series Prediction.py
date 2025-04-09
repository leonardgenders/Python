'''
M4 Assignment-1: RNN for Time Series Prediction.
This assignment is based on the Time Series Prediction lab 10.9.6 from ISLP Chapter 10.
Please use the textbook lab as a reference.
Note that the textbook lab is written using PyTorch.
You should write your model using Tensorflow.
The goal is to predict log_volume using lagged data.

Step 0: Load Libraries
Load the libraries you need in this lab
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

import random
import numpy as np
import tensorflow as tf
import pandas as pd

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

# import the necessary libraries
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# FREEZE CODE BEGIN
'''
Step 1: Load & Prep Dataframes
Load the NYSE dataset from the NYSE.csv file available in the file tree to the left.
The date column gives you the timestamps of the time Series.
The train column indicates True for records to be used in the train set, and False for those to be used in the test set.
For this step, let's keep only these 3 columns: 'DJ_return', 'log_volume', 'log_volatility'
Standardize all 3 columns using ScikitLearn's StandardScaler.
In the starter code given below:
  - cols is a list of the names of these 3 columns.
  - X is a dataframe that contains only these 3 columns from NYSE.csv.
**Q4-1-0 Print "0. the shape of datafrmae X: xxx" - Replace xxx with the proper value
**Q4-1-1 Print "1. the first record of dataframe X: xxx" - Replace xxx with proper values
'''
cols = ['DJ_return', 'log_volume', 'log_volatility']
# FREEZE CODE END

# Load the NYSE dataset from the NYSE.csv file 
NYSE = pd.read_csv('NYSE.csv')
cols = ['DJ_return', 'log_volume', 'log_volatility']
X = pd.DataFrame(StandardScaler(
                  with_mean = True,
                  with_std=True).fit_transform(NYSE[cols]),
                columns=NYSE[cols].columns,
                index=NYSE.index)
# view
# print(X)

# view first record


# **Q4-1-0 Print "0. the shape of datafrmae X: xxx" - Replace xxx with the proper value
print("0. the shape of datafrmae X: ", X.shape)

# **Q4-1-1 Print "1. the first record of dataframe X: xxx" - Replace xxx with proper values
print("1. the first record of dataframe X: ", X.iloc[0])

# FREEZE CODE BEGIN
'''
Use code from the textbook lab to set up lagged versions of these 3 data columns (using the starter code given to you here.)
Add column 'train' from the original dataset to the current dataframe X as the last column (to the right).
**Q4-1-2 Print "2. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
**Q4-1-3 Print "3. the first record of the data frame with lags: xxx" - Replace xxx with proper values
'''
# FREEZE CODE END

# Use code from the textbook lab to set up lagged versions of these 3 data columns (using the starter code given to you here
for lag in range(1, 6):
  for col in cols:
    newcol = np.zeros(X.shape[0]) * np.nan
    newcol[lag:] = X[col].values[:-lag]
    X.insert(len(X.columns), "{0}_{1}".format(col, lag), newcol)
X.insert(len(X.columns), 'train', NYSE['train'])

# **Q4-1-2 Print "2. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
print("2. the shape of dataframe X with lags: ", X.shape)

# **Q4-1-3 Print "3. the first record of the data frame with lags: xxx" - Replace xxx with proper values
print("3. the first record of the data frame with lags: ", X.iloc[0])

# FREEZE CODE BEGIN
'''
Drop any rows with missing values using the dropna() method.
**Q4-1-4 Print "4. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
**Q4-1-5 Print "5. the first record of dataframe X with lags: xxx" - Replace xxx with proper values
'''
# FREEZE CODE END

# Drop any rows with missing values using the dropna() method.
X = X.dropna()

# **Q4-1-4 Print "4. the shape of dataframe X with lags: xxx" - Replace xxx with the proper value
print("4. the shape of dataframe X with lags: ", X.shape)

# **Q4-1-5 Print "5. the first record of dataframe X with lags: xxx" - Replace xxx with proper values
print("5. the first record of dataframe X with lags: ", X.iloc[0])

# FREEZE CODE BEGIN
'''
Create the Y response target using the 'log_volume' column from dataframe X.
Extract the 'train' column from dataframe X as a separate variable called train. Drop the 'train' column from dataframe X.
Later on we will use the train variable to split the dataset into train vs. test.
Drop the current day’s DJ_return (the "DJ_return" column) and log_volatility from dataframe X.
- Current day refers to the non-lagged columns of these two variables. 
- In other words, remove these two X features, and also the Y response that came from dataframe X.
**Q4-1-6 Print "6. the first 3 records of the Y target : xxx" - Replace xxx with proper values.
**Q4-1-7: Print "7. the first 3 records of the train variable: xxx" - Replace xxx with proper values.
**Q4-1-8: Print "8. the first 3 records of dataframe X: xxx" - Replace xxx with proper values.
'''
# FREEZE CODE END

# Create the Y response target using the 'log_volume' column from dataframe X. Extract the 'train' column from dataframe X as a separate variable called train.
Y, train = X['log_volume'], X['train']
# Drop the 'train' column from dataframe X.
X = X.drop(columns=['train'] + cols)
# view
# print("X columns are:", X.columns)

# **Q4-1-6 Print "6. the first 3 records of the Y target : xxx" - Replace xxx with proper values.
print("6. the first 3 records of the Y target : ", Y.iloc[0:3])

# **Q4-1-7: Print "7. the first 3 records of the train variable: xxx" - Replace xxx with proper values.
print("7. the first 3 records of the train variable: ", train.iloc[0:3])

# **Q4-1-8: Print "8. the first 3 records of dataframe X: xxx" - Replace xxx with proper values.
print("8. the first 3 records of dataframe X: ", X.iloc[0:3])

# FREEZE CODE BEGIN
'''
To fit the RNN, we must reshape the X dataframe, as the RNN layer will expect 5 lagged versions of each feature as indicated by the (5,3) shape of the RNN layer below. 
We first ensure the columns of our X dataframe are such that a reshaped matrix will have the variables correctly lagged. 
We use the reindex() method to do this. 
The RNN layer also expects the first row of each observation to be earliest in time..
So we must reverse the current order.
Follow the textbook lab code to reorder/reindex the columns properly.
**Q4-1-9: Print "9. the first 3 records of X after reindexing: xxx" - Replace xxx with proper values.
'''
# FREEZE CODE END

# Follow the textbook lab code to reorder/reindex the columns properly
ordered_cols = []
for lag in range(5,0,-1):
  for col in cols:
    ordered_cols.append('{0}_{1}'.format(col, lag))
X = X.reindex(columns=ordered_cols)

# view
# print('X cols after reindexing: ', X.columns)

# **Q4-1-9: Print "9. the first 3 records of X after reindexing: xxx" - Replace xxx with proper values.
print("9. the first 3 records of X after reindexing: ", X.iloc[0:3])

'''
Reshape dataframe X as a 3-D Numpy array such that each record/row has the shape of (5,3). Each row represents a lagged version of the 3 variables in the shape of (5,3). 
**Q4-1-10: Print "10. the shape of X after reshaping: xxx" - Replace xxx with proper values.
**Q4-1-11: Print "11. the first 2 records of X after reshaping: xxx" - Replace xxx with proper values. 
''' 
# Reshape dataframe X as a 3-D Numpy array such that each record/row has the shape of (5,3). Each row represents a lagged version of the 3 variables in the shape of (5,3).
X_rnn = X.to_numpy().reshape((-1,5,3))

# **Q4-1-10: Print "10. the shape of X after reshaping: xxx" - Replace xxx with proper values.
print('10. the shape of X after reshaping: ', X_rnn.shape)

# **Q4-1-11: Print "11. the first 2 records of X after reshaping: xxx" - Replace xxx with proper values. 
print("11. the first 2 records of X after reshaping: ", X_rnn[0:2]) # no iloc bc now np array

'''
Now we are ready for RNN modeling.
Set up your X_train, X_test, Y_train, and Y_test using the X dataframe, Y response target, and the train variable you have created above.
Include records where train = True in the train set, and train = False in the test set.
Configure a Keras Sequential model with
(1) proper input shape,
(2) SimpleRNN layer with 12 hidden units, the relu activation function, and 10% dropout
(3) a proper output layer.
Do not name the model or any of the layers.
**Q4-1-12: Print a summary of your model. 
'''

# Set up your X_train, X_test, Y_train, and Y_test using the X dataframe, Y response target, and the train variable you have created above.
X_train = X_rnn[train] # true in the train set
X_test = X_rnn[~train] # false in the test set
Y_train = Y[train]
Y_test = Y[~train]


# Configure a Keras Sequential model with:
model = Sequential()
model.add(SimpleRNN(12, # 12 hidden units
          activation='relu', # relu activation function
          dropout=0.1, # 10% dropout
          input_shape = (5,3))) # input_shape worked, need _ between input shape
model.add(Dense(1, activation='linear')) # a proper output layer - linear because predicting one value (log volume)

# **Q4-1-12: Print a summary of your model. 
model.summary()

'''
Compile the modle with
(1) the adam optimizer,
(2) MSE as the loss,
(3) MSE as the metric.

Fit the model with
(1) 200 epochs,
(2) batch size of 32.
No need to print epoch-by-epoch progress.

There is nothing to print for this step.
'''

# Compile the modle with the adam optimizer, MSE as the loss, and MSE as the metric.
model.compile(optimizer='adam',
              loss='MSE',
              metrics=['MSE'])

# Fit the model with 200 epochs, and batch size of 32
estimate = model.fit(X_train,
                    Y_train,
                    epochs=200,
                    batch_size=32)

'''
Evaluate the model using model.evaluate() with the test set
**Q4-1-13 Print "13. Test MSE: xxx" - Replace xxx with the proper value.
'''

# Evaluate the model using model.evaluate() with the test set
score, MSE = model.evaluate(X_test,
                                Y_test,
                                batch_size=32)

# **Q4-1-13 Print "13. Test MSE: xxx" - Replace xxx with the proper value
print("13. Test MSE: ", MSE)
