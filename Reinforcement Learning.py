'''
M6 Assignment: Reinforcement Learning
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

# 1. Use 2 arms in this multi-armed bandit problem
n = 2 # number of arms
# Arm 0 has mean = 3 and sd = 1
# Arm 1 has mean = 6 and sd = 2

# 2. Create a numpy array called arms to store the mean and sd for each of the two arms. 
# Each record/row stores a pair of mean and sd for an arm. In other words, your arms array 
# should contain two records, one for each arm. Each record should be a [mean, sd] pair.
arms=np.array([[3,1], [6,2]])

# 3. Q6-0 Print the arms array. Format the output as “0: xxx” – Replace xxx with the arms array.
print("0: ", arms)

# 4. Define a function called reward to generate a reward score given an arm's mean and sd
# using the np.random.normal() function.
def reward(dist): # each dist is a tuple of mean & sd, one record
  mean = dist[0] # error here because receiving arms as array
  sd = dist[1]
  # The function then generates a random reward score using the np.random.normal() function
  # the given mean and sd
  zscore = np.random.normal(loc=0, scale=1, size=1) # loc is center of distro, scale is sd of distro, size = shape
  score = mean + zscore * sd
  # reward function returns the reward score as a single float value
  return float(score)

# 5. Q6-1 Print the reward score of mean = 5 and sd = 1 using the reward function from step 4. 
# Format the output as “1: xxx” – Replace xxx with the actual reward score.
print("1: ", reward([5,1]))

# 6. Initialize a numpy array as a memory array for storing action-value pairs.
# 7. Store just one record in this memory array as a starting value. Use arm 0 as the start arm
starting_arm=0
av = np.array([starting_arm, 0]).reshape(1,2) # starting reward of 0, output should be [[0 0]]

# 8. Q6-2 Print your memory array. Format the output as “2: xxx” – Replace xxx with the memory array.
print("2: ", av)

# 9. Define a greedy method to select best arm based on memory array.
# Define a function called bestArms that takes in the array of action-value pairs 
# and iterates through each record to find the action with the highest mean reward
def bestArm(a):
  bestArm = 0 # make arm zero the default best arm
  bestMean = 0
  for u in a: # iterate thru each record of the input memory array
    this_action = a[np.where(a[:, 0] == u[0])]
    avg = np.mean(this_action[:, 1]) # calc mean reward (expected reward) for each arm
    if bestMean < avg:
      bestMean = avg
      bestArm = int(u[0])
  return bestArm

# 10. Run the multi-arm bandit simulation 20 times with eps = 0.7 or 70% exploration and 30% exploitation
n_trials = 20
eps = .9

# Main loop for each play
for i in range(n_trials):
  lottery = random.random()
  if lottery > eps: # exploitation
    choice = int(bestArm(av)) # retrieve the best arm from bestArm function
  else: # exploitation
   choice=np.random.randint(0, n) # use the np.random.randint() function to chooce one of the two arms as best
  score = reward(arms[choice])
  thisAV = np.array([choice, score]).reshape(1,2) # choice, reward
  av = np.concatenate((av, thisAV), axis=0) # add to our action-value memory array
  # calc mean reward
  runningMean = np.mean(av[:,1])
  # Q6-3 Print the cumulative mean reward after each trial. Format the output as “3: xxx” – Replace xxx with the cumulative mean per trial. 
  print("3: ", runningMean)

# Q6-4 Print the memory array after all trials are completed.
print(av)

# Cumulative reward test
print(f'Cumulative reward: {np.sum(av[:,1])}')
# 0.7 result: 95.11430646539108 HIGHEST
# 0.2 result: 63.00772310567827
# 0.9 result: 78.96273670043823
