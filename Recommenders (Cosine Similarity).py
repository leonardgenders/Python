'''
M5 Lab-1: Cosine Simularity
Compute cosine similarity using dot product and vector norm.
You will need to use numpy.dot and numpy.linalg.norm.
Please use the Numpy library only. Do NOT use any other library.
Write a function that takes any two of the following np arrays as input,
and compute their cosine similarity as output.
'''

# FREEZE CODE BEGIN
user1 = [1, 1, -10, 0]
user2 = [4, 5, 6, 8]
user3 = [4, 5, 6, 9]
user4 = [4, 5, 6, 10]

movie1 = [-1, 1, -10]
movie2 = [1, 2, 3]
movie3 = [1, 2, 4]
movie4 = [30, -2, 100]
# FREEZE CODE END

'''
Print the following
**Q5-1-0 Print** "user1 and user2: xxx" - Replace xxx with cosine similarity of user1 and user2
**Q5-1-1 Print** "user2 and user4: xxx" - Replace xxx with cosine similarity of user2 and user4
**Q5-1-2 Print** "user3 and user4: xxx" - Replace xxx with cosine similarity of user3 and user4
**Q5-1-3 Print** "movie1 and movie2: xxx" - Replace xxx with cosine similarity of movie1 and movie2
**Q5-1-4 Print** "movie2 and movie3: xxx" - Replace xxx with cosine similarity of movie2 and movie3
**Q5-1-5 Print** "movie3 and movie4: xxx" - Replace xxx with cosine similarity of movie3 and movie4
'''

# Import the necessary packages
import numpy as np
from numpy.linalg import norm

# Write a function that takes any two of the following np arrays as input, and computer their cosine similarity as output
def similarity(array1, array2):

  # calculate dot product first
  dot = np.dot(array1, array2)

  # calculate cosine values
  cos = np.dot(array1, array2)/(norm(array1)*norm(array2))

  return cos

# **Q5-1-0 Print** "user1 and user2: xxx" - Replace xxx with cosine similarity of user1 and user2
print("user1 and user2: ", similarity(user1, user2))

# **Q5-1-1 Print** "user2 and user4: xxx" - Replace xxx with cosine similarity of user2 and user4
print("user2 and user4: ", similarity(user2, user4))

# **Q5-1-2 Print** "user3 and user4: xxx" - Replace xxx with cosine similarity of user3 and user4
print("user3 and user4: ", similarity(user3, user4))

# **Q5-1-3 Print** "movie1 and movie2: xxx" - Replace xxx with cosine similarity of movie1 and movie2
print("movie1 and movie2: ", similarity(movie1, movie2))

# **Q5-1-4 Print** "movie2 and movie3: xxx" - Replace xxx with cosine similarity of movie2 and movie3
print("movie2 and movie3: ", similarity(movie2, movie3))

# **Q5-1-5 Print** "movie3 and movie4: xxx" - Replace xxx with cosine similarity of movie3 and movie4
print("movie3 and movie4: ", similarity(movie3, movie4))
