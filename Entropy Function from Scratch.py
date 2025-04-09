'''
1. Write the entropy function for binary classification from scratch using only Python's math module. Do NOT use Numpy.
2. Do NOT call a built-in entropy function from any Python library.
3. Use the math library only. Do not use any other library.
4. You have to write the entropy function from scratch on your own. Name it MyEntropy.
5. Your entropy function should take a probability value for one of the two classes as input, and output its entropy value.
6. Make sure the frozen starter code  runs properly without error and produces correct output.
'''

import math
def MyEntropy(p):
  # need to account for probabilities
  # -(p)log(base2)(p) - qlog(base2)q where p is initial prob and q is 1-p
  q = 1-p
  return - float((p*math.log(p, 2)) + (q*math.log(q, 2))) # returns actual values now and not None, need to add because sum

# test
# MyEntropy(.2)

# FREEZE CODE BEGIN
output0 = MyEntropy(.2)
print("The entropy value of probability .2 is " + str(output0)) # Q4-0-0

output1 = MyEntropy(.8)
print("The entropy value of probability .8 is " + str(output1)) # Q4-0-1

output2 = MyEntropy(.5)
print("The entropy value of probability .5 is " + str(output2)) # Q4-0-2
# FREEZE CODE END
