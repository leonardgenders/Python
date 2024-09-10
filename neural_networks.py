''' 
Assignment1-1: Deep Learning (DL) Regressor
Step 0: Load Libraries
'''
import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

'''
Step 1: Load & Prep Dataframes
We will use the 'wine quality with category value.xlsx' data file available in the file tree to the left
Load the dataset as a pandas dataframe
Explore the data dictionary using the Reference link in the right-side panel
We will build a model to predict "quality" as a numeric target

Create the x dataframe using everything except for "quality"
**Q1-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe 

Create the y target using the "quality" column
**Q1-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the y target
'''
# load the dataset as a pandas dataframe
DF = pd.read_excel('wine quality with category value.xlsx')
# DF.info()
# repository states no missing values

# Create the x dataframe using everything except for "quality"
x = DF.copy()
x.drop(['quality'], axis=1, inplace=True)
x.info() # check quality is dropped
x.head()

# **Q1-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe
print(x[15:18:1])

# Create the y target usingthe "quality" column
y = DF["quality"]
y.info()

# **Q1-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the y target
print(y[15:18:1])
'''
Step 2: Prep train vs. test sets
Use train_test_split from sklearn to split the x dataframe and y target from the previous step (after one-hot coding)
50% training and 50% for testing
Use the random seed 1693 in the train_test_split statement
** Q1-1-2 Print** the first record in the y test set
'''

# red wine is 1 and white wine is 0 
# Use the train_test_split from sklearn to split the x dataframe and the y target from the previous step (after one-hot encoding)
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.5,
                                                    random_state=1693) # 50% train 50% test

# ** Q1-1-2 Print** the first record in the y test set
# print('y_test first record result:')
print(y_test.iloc[0]) # result is 5
'''
Step 3: Train Model
Create a Sequential model and name it "WineQuality"
Add one Dense hidden layer with 
(1) 7 nodes 
(2) an adequate number for input dimension, 
(3) the relu activation function, and
(4) layer name "HL1"
Add another Dense hidden layer with 
(1) 5 nodes 
(3) the relu activation function, and
(4) layer name "HL2"
Add one Dense output layer with 
(1) an adequate number of nodes
(2) an adequate activation function, and
(3) layer name "OL"
**Q1-1-3: Print** a summary of your model using model.summary()
'''
# Create a Sequential model and name it "WineQuality"
model = Sequential(name="WineQuality")
# Add one Dense hiddne layer with:
model.add(Dense(units=7, # 7 nodes
                input_dim=12, #12 dimensions
                activation='relu', #relu activation function
                name='HL1')) #layer name HL1
# Add another Dense hidden layer with:
model.add(Dense(units=5, #5 nodes
                activation='relu', #relu activation function
                name='HL2'))
# Add one Dense output layer with:
model.add(Dense(1, # predicting a single value for y_hat
                activation='linear', #linear activation for regressor
                name='OL'))
# **Q1-1-3: Print** a summary of your model using model.summary()
model.summary()

'''
Compile the model with 
(1) the mse as the loss function,
(2) adam as the optimizer,
(3) mse as the metrics

Train the model with:
(1) the train set from above,
(2) the test set split with 25% for validation,
(3) 150 epochs
(4) no prinout of each epoch result
There is nothing to print for this step
'''
# Compile the model with:
model.compile(loss='mean_squared_error', # mse as the loss function
              optimizer='adam', # adam as the optimizer
              metrics=['mse']) # mse as the metrics

# Train the model with:
estimate = model.fit(X_train, # the train set from above, no verbose bc no print
                    y_train, # y target = quality
                    epochs=150, # 150 epochs
                    validation_split=.25) # reserve last 25% of train set as valdation set

'''
Step 4: Evaluate Model Performance
**Q1-1-4 Print** the predicted values for the first two records
**Q1-1-5 Print** the model mse value from the last epoch
'''
# **Q1-1-4 Print** the predicted values for the first two records
preds = model.predict(X_test)
print(preds[0:2])

# **Q1-1-5 Print** the model mse value from the last epoch
print(estimate.history['mse'][-1])

''' 
Lab1-1: DL Regressor
Step 0: Load Libraries
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

'''
Step 1: Load & Prep Dataframes
Q1-1-0
We will use the Auto MPG dataset
Download the dataset from http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
Replace all missing values with zeros
Use cylinders, horsepower and weight (in this exact order) as predictors
We will predict mpg as the y target
**Q1-1-0 Print** the mean of y (that is, mean mpg) - use Numpy
**Q1-1-0 Print** the variance of y in a new line - again use Numpy
'''
# load the Auto MPG dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
DF = pd.read_csv(url, names=column_names, na_values='?', comment='\t', 
  sep=' ', skipinitialspace=True)
# view DF
print(DF.info())

# check number of NAs/missing values
print('Missing Values')
print(DF.isna().sum()) # missing in horsepower
DF['horsepower'] = DF['horsepower'].replace(np.nan, '0').astype(float)

# create the predictors
predictors = DF[['cylinders', 'horsepower', 'weight']]
print(predictors)

# create the target
target = DF['MPG'] # single bracket makes a pandas series with no column
print(target)

# **Q1-1-0 Print** the mean of y (that is, mean mpg) - use Numpy
y = np.mean(target)
print(y)

# **Q1-1-0 Print** the variance of y in a new line - again use Numpy
y_var = np.var(target)
print(y_var)
'''
Step 2: Prep train vs. test sets - Skip this step
'''

'''
Step 3: Train Model
Q1-1-1
Create a Sequential model and name it "MyFirstNN"
Add one Dense hidden layer with 
(1) 5 nodes 
(2) an adequate number for input dimension, 
(3) the relu activation function, and
(4) layer name "MyFirstHL"
Add one Dense output layer with 
(1) an adequate number of nodes
(2) an adequate activation function, and
(3) layer name "MyFirstOL"
**Q1-1-1: Print** a summary of your model using model.summary()
Compile the model with 
(1) entire x and y - no train/test split for now,
(2) the mse as the loss function,
(3) adam as the optimizer,
(4) mse as the metrics
Fit the model with
(1) entire x and y - no train/test split for now,
(2) 100 epochs
(3) no prinout of each epoch
'''
# Q1-1-1
# Create a Sequential model and name it "MyFirstNN"
model = Sequential(name="MyFirstNN")
# Add one Dense hidden layer
model.add(Dense(units=5, # 5 nodes
                input_dim=3, # adequate number for input (3 - cyl, horse, wt)
                activation='relu', # relu activation function
                name='MyFirstHL')) # layer name "MyFirstHL"
model.add(Dense(units=1, # adequate number of nodes (1 for regression model)
                activation='linear', # adequate activation function (always linear for regression)
                name='MyFirstOL'))
# **Q1-1-1: Print** a summary of your model using model.summary()
model.summary()

# Compile the model with
model.compile(loss='mean_squared_error', # mse as the loss function
              optimizer='adam', # adam as the optimizer
              metrics=['mse']) # mse as the metrics

# Fit the model with
estimate = model.fit(predictors, 
                    target, 
                    epochs=100,
                    verbose=0) # entire x and y, no train/test split and 100 epochs, no printout of each epoch

'''
Q1-1-2
Step 4: Evaluate Model Performance
**Print** the predicted values for the first two records
**Print** the model mse value from the last epoch
'''
# Q1-1-2
# Step 4: Evaluate Model Performance
# **Print** the predicted values for the first two records
preds=model.predict(predictors)
print(preds[0:2])

# **Print** the model mse value from the last epoch
print(estimate.history['mse'][-1])


# Wine Quality Combined Classification Model
'''
Step 0: Load Libraries
The starter code given below only imports some, not all, of the libraries you need.
You should load the other libraries you need below.
'''
# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import sklearn as sk
from sklearn.model_selection import train_test_split

'''
Step 1: Load & Prep Dataframes
We will use the 'wine quality combined.xlsx' data file available in the file tree to the left.
Load the dataset as a pandas dataframe.
Explore the data dictionary using the Reference link in the right-side panel.
We will build a model to predict "quality" as a categorical target.

One-hot code categorical variables within the dataframe using the get_dummies() function from Pandas.
Reference the Pandas documentation to see how to use get_dummies() properly.
Treat "quality" and "category" as categorical variables (meaning the numbers indicate categories, even though the variables are integers.)
Note that you can decide when to one-hot code each of the variables,
and whether to do so together or separately,
as long as your printouts below are correct.

Create the x dataframe using everything except for "quality."
**Q2-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe after one-hot coding from the previous step.

Create the y target using the one-hot coded "quality" columns.
**Q2-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) from the one-hot coded y target
'''

# Load the dataset as pandas dataframe
DF = pd.read_excel('wine quality combined.xlsx')
print(DF.head())# view

# One-hot code categorical variables witihin the dataframe using the get_dummies() function from Pandas
# Treat 'quality' and 'category' as categorical variables
# Starting with 'quality'
y_raw = DF['quality']

# Check the unique values of 'quality' as y_raw
unique_elements, counts_elements = np.unique(y_raw, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# Convert quality using get_dummies()
y = pd.get_dummies(y_raw)
print(y_raw.iloc[:10]) # original values
print(y[:10]) # get_dummies()

# Now doing 'category'
# Create the x dataframe using everything except for "quality."
x_raw = DF.copy()
x_raw.drop(['quality'], axis=1, inplace=True)

# Convert just the 'category' column using get_dummies() and the columns argument
x = pd.get_dummies(x_raw, columns=['category'])
print(x_raw.iloc[:10]) # original values
print(x[:10]) # get_dummies creates category_red and category_white cols

# **Q2-1-0 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) of the x dataframe after one-hot coding from the previous step.
print(x[15:18:1])

# **Q2-1-1 Print** 16ths, 17th, and 18th records (i.e., starting with row index 15) from the one-hot coded y target
print(y[15:18:1])

'''
Step 2: Prep train vs. test sets
Use train_test_split from sklearn to split the x dataframe and y target from the previous step (after one-hot coding).
50% training and 50% for testing
Use the random seed 1693 in the train_test_split statement.
** Q2-1-3 Print** the first record in the y test set
** Tip: Python uses zero-based, not one-based, indexing. **
'''

# Use train_test_split from sklearn to split the x dataframe and y target from the previous step (after one-hot coding)
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.5, # 50% training and 50% for testing
                                                    random_state=1693) # random seed 1693 

# ** Q2-1-3 Print** the first record in the y test set
# print('y_test first record result:')
print(y_test.iloc[0])

'''
Step 3: Train model
Create a Sequential model with no name
Add one Dense hidden layer with
(1) 15 nodes,
(2) an adequate number for input dimension,
(3) the tanh activation function, and
(4) no layer name
Add a second Dense hidden layer with
(1) 20 nodes,
(3) the relu activation function, and
(4) no layer name
Add one Dense output layer with
(1) an adequate number of nodes
(2) the appropriate activation function, and
(3) no layer name
**Q2-1-3 Print** a summary of your model using model.summary()
'''

# Create a Squential model with no name
model = Sequential()
# Add one Dense hidden layer with:
model.add(Dense(units=15, # 15 nodes
                input_dim=13, # 13 dimensions after one-hot coding
                activation='tanh')) # tanh activation with no layer name
# Add a second Dense hidden layer with:
model.add(Dense(units=20, # 20 nodes
                activation='relu')) # relu activation function, no layer name
# Add one Dense output layer with:
model.add(Dense(units=7, # try 7 classes from unique values above
                activation='softmax')) # multi-class, no name

# **Q2-1-3 Print** a summary of your model using model.summary()
model.summary()

'''
Compile the model with
(1) CategoricalCrossentropy as the loss function,
(2) sgd as the optimizer
(3) accuracy as the metric

Fit the model with
(1) the train set you created in Step 2, and do NOT use the test set you created in Step 2 as validation_data.
(3) 100 epochs
(4) no prinout of each epoch

Do not specify learning rate, momentum, or other parameters that are not required in the instructions.

There's nothing to print for this part
'''

# Compile the model with:
model.compile(loss='CategoricalCrossentropy', # CategoricalCrossentropy as the loss function
              optimizer='sgd', # sgd as the optimizer
              metrics=['accuracy']) # accuracy as the metric

# Fit the model with:
estimate = model.fit(X_train,
                    y_train,
                    epochs=100) # 100 epochs and no printout of each epoch

'''
Step 4: Evaluate Model Performance
**Q2-1-4 Print** test accuracy using the evaluate() function from Keras, and the test set you created in step 2.
         Format the printout as 'test_acc: .5'
         Replace .5 with the correct value
Convert predicted probabilities into predicted classes
CAUTION: Think carefully about how get_dummies() works and how you should structure the conversion so you predict the correct class labels.
**Q2-1-5 Print** predicted quality category (i.e. quality labels) for the 16ths, 17th, and 18th records from the original dataset.
**Q2-1-6 Calculate and print** the model's accuracy rate using the entire original dataset.
'''

# **Q2-1-4 Print** test accuracy using the evaluate() function from Keras, and the test set you created in step 2.
# Format the printout as 'test_acc: .5'
# Replace .5 with the correct value
results = model.evaluate(X_test, y_test)
print(f'test_acc: {results[1]}')

# Convert predicted probability into predicted classes
pred_prob = model.predict(x)
print(pred_prob)
# Convert probabilities into classes using argmax
pred_class = pred_prob.argmax(axis=-1)

# **Q2-1-5 Print** predicted quality category (i.e. quality labels) for the 16ths, 17th, and 18th records from the original dataset.
print(pred_class[15:18])

# **Q2-1-6 Calculate and print** the model's accuracy rate using the entire original dataset.
# print(sum(pred_class == y_raw)/len(pred_class)) # output is way off, showing .00400
# issue with y_raw? not one-hot coded
# print(y.info()) # try y
# print(pred_class)

print(f'y.values: {y.values}') # view before argmax
y_vals = y.values.argmax(axis=-1) # this should put in same format as pred_class
print(f'y_vals after argmax attempt: {y_vals}')

# try to get a better accuracy than before
# print('accuracy after using argmax attempt on y values instead of y_raw format:')
# **Q2-1-6 Calculate and print** the model's accuracy rate using the entire original dataset.
print(sum(pred_class == y_vals)/len(pred_class)) #output is 0.437 which is closer to the epochs accuracy
