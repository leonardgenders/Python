'''
Implement the multi-armed bandit problem
Do NOT format any of your printed values
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

import random
import numpy as np

random.seed(1693)
np.random.seed(1693)
# FREEZE CODE END

'''
Specify the number of arms as 5
Generate a Python list of random probabilities (values between 0 and 1) for each arm using np.random.rand().
**Q6-0 Print the list of probabilities
'''

# Specify the number of arms as 5
n = 5
# List of random probabilites for each arm (5) using np.random.rand()
arms = (np.random.rand(n))

# **Q6-0 Print the list of probabilities
print(arms)

'''
Use this function to define the environment (world) that generates the reward function.
This model of the environment is not visible to the agent
This reward function simulates randomly generated reward for an arm (with a given prob)
This function accepts a probability value as an input argument
And outputs a total reward value 
Iterate through each arm and generate a random value for the arm
If the random value exceeds the arm's probability, then you'll add a 1 to the total reward
After going through all of the arms, the reward function would return the total reward value
'''

# FREEZE CODE BEGIN
def reward(prob): # prob is the probability of a given arm
    n_iterations = 10
    reward = 0
    for i in range(n_iterations):
        if random.random() > prob:
            reward += 1
    return reward
# FREEZE CODE END

'''
Use the reward function defined above to generate a total reward value for each of the arms
**Q6-1 Print "The total reward for arm x is yyy" - Replace x with arm index, and yyy with the actual reward value
         The first printout should be for arm 0
'''

# Use the reward function defined above to generate a total reward value for each of the arms
# **Q6-1 Print "The total reward for arm x is yyy" - Replace x with arm index, and yyy with the actual reward value
x = 0
for a in arms:
    print(f'The total reward for arm {x} is {reward(a)}')
    x += 1
    

'''
Use the code snippet below to initialize a memory array for storing action-value pairs
memory array stores history of all actions & rewards
row = an index reference to the arms array (1st element), and the reward received (2nd element)
Example: row x = [2, 8] means
action 2 was taken (the 3rd element in our arms array)
you received a reward of 8 for taking that action
has 1 row defaulted to random action index

Store just one record in this memory array as a starting value
Use arm 3 as the starting arm
**Q6-2 Print your memory array
'''

# FREEZE CODE BEGIN
starting_arm=3
av = np.array([starting_arm, 0]).reshape(1,2)
# FREEZE CODE END

# **Q6-2 Print your memory array
print(av)

'''
Define greedy method to select best arm based on memory array
The bestArm() function should have one input argument to receive a memory array (e.g., the one defined above)
Make arm zero the default best arm.
Iterate through each record of the input memory array.
Calculate mean reward (expected reward) for each arm.
Return the arm index as an integer with the highest mean reward - this would be the best arm result

**Q6-3 Print the bestArm output using your memory array as input.
         Format the output as "The best arm is #x" - Replace x with the proper integer.
'''

# Define greedy method to select best arm based on memory array
# The bestArm() function should have one input argument to receive a memory array (e.g., the one defined above)
def bestArm(a):
    bestArm = 0 # Make arm zero the default best arm
    bestMean = 0
    for u in a: # iterate through each record of the input memory array
        # u[0] = action ID (arm 0 - 4)
        # u[1] = reward (value)
        this_action = a[np.where(a[:, 0] == u[0])] # Select all records with action u[0] (column 0) and save them into a new matrix called "this_action"
        avg = np.mean(this_action[:, 1]) # Calc mean reward for this action (reward val is in column 1)
        # print('avg = '+str(avg))
        if bestMean < avg:
            bestMean = avg
            bestArm = u[0]
    return bestArm

# **Q6-3 Print the bestArm output using your memory array as input
output = bestArm(av)
Format the output as "The best arm is #x" - Replace x with the proper integer.
print(f'The best arm is #{output}')

'''
Run the multi-arm bandit simulation 10 times with the epsilon-greedy policy -
for 75% of exploitation and 25% of exploration.
For an exploitation trial, use the bestArm function to find the best arm, and get the best arm's reward by calling the reward function.
For an exploration trial, pick an arm at random and get its reward by calling the reward function.
**Q6-4 Print the cumulative mean reward after each trial
'''

# Run the multi-arm bandit simulation 10 times with the epsilon-greedy policy
n_trials = 10
eps = .25 # 25% exploration and 75% exploitation
for i in range(n_trials):
    # For an explotation trial, use the bestArm function to find the best arm, and get the best arm's reward by calling the reward function
    if random.random() > eps: # exploitation
        choice = bestArm(av) # retrieve the best arm
        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0) # add the new trial to my av history/memory array
    else: # exploration
        choice = np.where(arms == np.random.choice(arms))[0][0] # pick a random arm
        thisAV = np.array([[choice, reward(arms[choice])]]) # choice, reward
        av = np. concatenate((av, thisAV), axis=0) # add to our action-value memory array
    # calc the mean reward
    
    runningMean = np.mean(av[:,1]) # calculate average of rows so far in column 1 (reward value)
    # **Q6-4 Print the cumulative mean reward after each trial
    print(runningMean)
