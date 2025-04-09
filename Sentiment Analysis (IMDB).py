'''
M4 Lab-2: LSTM for IMDB sentiment analysis
This lab is based on ISLP Chapter 10.
The objective is to correctly classify reviews into positive vs. negative ones.
Please first review the IMDB dataset documentaion on Keras.

Step 0: Load Libraries
Load the libraries you need in this lab
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

'''
Step 1: Load & Prep Dataframes

[Hyperparameter Tuning Zone]
It's good practice to keep important hyperparameters all in one place
So we can tune them easily later
The default value of 0 is provided here
You need to change these hyperparameter values here based on coding instructives given below
There is nothing to print in this section

CAUTION: Hyperparameter values are set low here to ensure the lab runs properly on Codio.
         On Colab you should feel free to try different values to improve model performance.
'''

max_features = 5000 # keep only 5000 most frequent words
maxlen = 50 # So the sequences are of the same lengths of 50
embedding_size = 0
batch_size = 256 # batch size 256
num_epochs = 5
lstm_units = 26
dropout = 0.1

# import necessary libararies
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.layers import LSTM


'''
Load the IMDB dataset from Keras using the dataset module
Keep only 5000 most frequent words
Reminder: Set such hyperparameter value in the [Hyperparameter Tuning Zone] above
Split the dataset into train vs. test automatically while reading in the dataset
We will refer to each record (row) as a sequence
**Q4-2-0 Print "xxx sequences in x_train" - Replace xxx with the size of the train sequences.
**Q4-2-1 Print the first 12 elements in the first sequence in a new line
**Q4-2-2 Print "First y value is xxx" - Replace xxx with the first element in y test in a new line.
'''
# Load the IMDB dataset from Keras using the dataset module
from tensorflow.keras.datasets import imdb

# Split the dataset into train vs. test automatically while reading in the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# We will refer to each record (row) as a sequence
# **Q4-2-0 Print "xxx sequences in x_train" - Replace xxx with the size of the train sequences
print(len(x_train), "sequences in x_train")

# **Q4-2-1 Print the first 12 elements in the first sequence in a new line
print(x_train[0][:12]) # first 12 words in the first document, vector of numbers between 0-9999, referring to words in the dictionary

# **Q4-2-2 Print "First y value is xxx" - Replace xxx with the first element in y test in a new line.
print("First y value is", y_test[0])

'''
Step 2: Prep train vs. test sets
Pad the sequences with zeros AFTER each sequence in both train and test sets
So the sequences are of the same lengths of 50
**Q4-2-3 Print "x_train shape: xxx" - Replace xxx with the shape of x train set
**Q4-2-4 Print "x_test shape: xxx" - Replace xxx with the shape of x test set
**Q4-2-5 Print "y_train shape: xxx" - Replace xxx with the shape of y train set
**Q4-2-6 Print "y_test shape: xxx" - Replace xxx with the shape of y test 
'''
# Pad the sequences with zeros AFTER each sequence in both train and test sets
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

# **Q4-2-3 Print "x_train shape: xxx" - Replace xxx with the shape of x train set
print("x_train shape: ", x_train.shape)

# **Q4-2-4 Print "x_test shape: xxx" - Replace xxx with the shape of x test set
print("x_test shape: ", x_test.shape)

# **Q4-2-5 Print "y_train shape: xxx" - Replace xxx with the shape of y train set
print("y_train shape: ", y_train.shape)

# **Q4-2-6 Print "y_test shape: xxx" - Replace xxx with the shape of y test 
print("y_test shape: ", y_test.shape)

'''
Step 3: Train model
Build a Sequential() model with 
1. A dense embedding layer with the appropriate input dimension, and 36 as the output dimension of embedding
2. An LSTM layer with the tanh activation function, 26 output units, and a 10% dropout rate
3. A dense output layer with the sigmoid activation function and an appropriate output dimension
Do NOT name any of the layers
**Q4-2-7 Print your model's summary
'''
# Build a Sequential() model with:
model = Sequential()
# 1. A dense embedding layer with the appropriate input dimension, and 36 as the output dimension of embedding
model.add(Embedding(input_dim=5000,
          output_dim=36))
# 2. An LSTM layer with the tanh activation function, 26 output units, and a 10% dropout rate
model.add(LSTM(lstm_units,
              activation='tanh',
              dropout=dropout))
# 3. A dense output layer with the sigmoid activation function and an appropriate output dimension
model.add(Dense(1, activation='sigmoid'))

# **Q4-2-7 Print your model's summary
model.summary()

'''
CAUTION: This step is the most computationally intensive
         Feel free to comment out this code segment when you're running Check It buttons for previous steps
Compile your model with 
- binary_crossentropy as the loss function
- adam as the optimizer
- accuracy as the metrics

Train your model with 
- the train and test sets you created above
- batch size of 256
- 5 epochs (to minimize demand for computing resources)
Turn off printing epoch outputsm so it's easier for you to inspect the output

Step 4: Evaluate model performance
Use the history property of the model fit output
**Q4-2-8 Print** training accuracy
**Q4-2-9 Print** validation accuracy    
'''

# Compile your model with binary_crossentropy as the loss function, adam as the optimizer, accuracy as the metrics
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train your model with the train and test sets you created above, batch size of 256, 5 epochs (to minimize demand for computing resources)
# Turn off printing epoch outputsm so it's easier for you to inspect the output
estimate = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_data=[x_test, y_test]) # need val for estimate.history['val_accuracy'] - maybe this spot to fix?

# Use the history property of the model fit output
score, accuracy = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size)

# **Q4-2-8 Print** training accuracy
print(estimate.history['accuracy'])

# **Q4-2-9 Print** validation accuracy
print(estimate.history['val_accuracy'])
