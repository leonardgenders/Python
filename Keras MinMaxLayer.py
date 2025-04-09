''' 
M5 Assignment-0: Create and use a subclass of the keras.layers.Layer class
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
# Import additional lirbaries/modules as needed

# FREEZE CODE END

# import the necessary lirbaries
import keras
from keras import layers


# Create a subclass of keras.layers.Layer called MinMaxLayer that
class MinMaxLayer(layers.Layer):
  
  def __init__(
    self,
    units = 12, # default value of 12 units
    input_dim = 6): # has a default value of 6 for input_dim
  
    ## a) inherits from the keras.layers.Layer class
    super().__init__()

    # set up units and input_dim activation representations
    self.units = units
    self.input_dim = input_dim

  # Replace the inherited 'call method' with a new one which, with an input 1-dimensional tensor, will return
  def call(self, array):

    if len(array) < self.input_dim:
      print(f'Error when checking input: expected to have {self.input_dim} as input_dim, but got array with {len(array)} elements')

    else: # using tf.math.reduce_max and .reduce_min
      scaled = ((array - tf.math.reduce_min(array))/(tf.math.reduce_max(array)-tf.math.reduce_min(array))) # minmaxscaler() function formula
      return scaled        


y1 = np.array([2,4,6,8,10])
y2 = np.array([5,10,15])

# Create a MinMaxLayer object and name it MyFirstLayer1
MyFirstLayer1 = MinMaxLayer(input_dim=5) # 5 elements in y1, define input_dim changes here and not in output variable below or get typeerror

output = MyFirstLayer1(y1)

# Q5-0-0 Print the output of MyFirstLayer1 using the array y1 as the input and the correct number
# of input dimensions (5)
# current result: tf.Tensor([0.   0.25 0.5  0.75 1.  ], shape=(5,), dtype=float64) between 0-1
print(output)

# Create a MinMaxLayer object and name it MySecondLayer2
MySecondLayer2 = MinMaxLayer() # y2 is input and has 3 elements

# Q5-0-1 Print the output of MySecondLayer2 using the array y2 as input and the default number of input dimensions
output2 = MySecondLayer2(y2)
print(output2) # returns Error statement from above
# 
# output from example x array from instructiions: tf.Tensor([0.  0.2 0.4 0.6 0.8 1. ], shape=(6,), dtype=float64)

print(str(MyFirstLayer1(np.array([-2,-1,0,2]))))
