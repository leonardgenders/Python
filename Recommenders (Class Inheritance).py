'''
M5 Lab-0: Class Inheritance
Use the Numpy library only. Do not use any other library.
Do NOT format any of your print outputs.
Create a class called Perceptron with
(1) 2 attributes: inputValue and activation
   inputValue allows one input feature x's value (numeric), with a default value of 0
   activation allows the user to specify the activation they want as a string variable, with the default of 'sigmoid'
   the user should be able to specify 'sigmoid', or 'relu'.

(2) 1 method: output 
   the ouput method will print the output of the user-specified activation function applied to the inputValue
   You should implement 2 options for this method:
     If the user specifies "relu" as the activation function, then the method would return the relu output of inputValue
     If the user specifies "sigmoid" as the activation function, then the method would return the sigmoid output of inputValue
     If the use specifies anything else, the method would return -999
'''
# import numpy as np
import numpy as np

# Create a class called Perceptron with 2 attributes: inputValue and activation
class Perceptron():

   # this constructor method defines what happens when a Perceptron is initiatied
   def __init__(self,
               inputValue = 0, # input attribute that allows one input feature x's value with default of 0
               activation = 'sigmoid'): # activation allows the user to specify the activation they want as a string variable with default as 'sigmoid' 
   # set up inputValue and activation representations
      self.inputValue = inputValue
      self.activation = activation

   # output method will print the output of the user-specified activation function applied to the inputValue
   def output(self):

      if self.activation == 'sigmoid':
         # If user specifies "sigmoid" as the activation function, then the method would return the sigmoid output of inputValue
         sigmoid_val = 1/(1 + np.exp(-self.inputValue))
         return sigmoid_val

         # If user specifies "relu" as the activation function, then the method would return the relu output of inputValue
      elif self.activation == 'relu':
         if self.inputValue > 0:
            return self.inputValue
         else:
            return 0

      # Else return -999   
      else:
         return -999

# test area
# Create an object of the class Perceptron and name it MyFirstNeuron
# MyFirstNeuron = Perceptron(-1, 'relu')
# **Q5-0-0 Print "MyFirstNeuron's output is xxx" - Replace xxx with the output of MyFirstNeuron using its output() method
# print("MyFirstNeuron's output is ", MyFirstNeuron.output()) # need to call the user defined function 'output'
# output is MyFirstNeuron's output is  <__main__.Perceptron object at 0x7013fbf0a380> when I don't use the output function


'''
Create an object of the class Perceptron and name it MyFirstNeuron
MyFirstNeuron should have an input value of -1, and relu as the activation function
**Q5-0-0 Print "MyFirstNeuron's output is xxx" - Replace xxx with the output of MyFirstNeuron using its output() method
'''

# Create an object of the class Perceptron and name it MyFirstNeuron
MyFirstNeuron = Perceptron(-1, 'relu')

# **Q5-0-0 Print "MyFirstNeuron's output is xxx" - Replace xxx with the output of MyFirstNeuron using its output() method
print("MyFirstNeuron's output is ", MyFirstNeuron.output())

'''
Create an object of the class Perceptron and name it HiddenLayerNode, with inputValue of 3, and the default activation function
**Q5-0-1 Print "HiddenLayerNode's output is xxx" - Replace xxx with the output of HiddenLayerNode using its output() method
'''

# Create an object of the class Perceptron and name it HiddenLayerNode, with inputValue of 3, and the default activation function
HiddenLayerNode = Perceptron(3) # default means not adding the type of activation function

# **Q5-0-1 Print "HiddenLayerNode's output is xxx" - Replace xxx with the output of HiddenLayerNode using its output() method
print("HiddenLayerNode's output is ", HiddenLayerNode.output())
