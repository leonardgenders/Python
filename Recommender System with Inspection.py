''' 
M5 Assignment-2: Create and inspect a recommender system
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

# import necessary lirbaries
from typing import Dict, Text
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# 1. Load the movielens/latest-small-ratings train dataset.
ratings = tfds.load('movielens/latest-small-ratings', split='train')

# 2. Load the movielens/latest-small-movies train dataset.
movies = tfds.load('movielens/latest-small-movies', split='train')

# 3. Select movie title and user id from the ratings dataset.
def transform_ratings(x):
  return { # return a dictionary with two keys
    'movie_title': x['movie_title'],
    'user_id': x['user_id']
  }
ratings = ratings.map(transform_ratings)

# 4. Select movie title from the movies dataset.
def transform_movies(x):
  return x['movie_title'] # returns the movie title

movies = movies.map(transform_movies)

# view / test
# print(ratings)
# print(movies)

# 5. Recreate the MovieLensModel class from M5 Lab - copy/pasted
class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)

# 6. Build vocabulary to convert user ids from the ratings dataset into integer indices for embedding layers.
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
# Adapt the object to the user_id feature of the ratings dataset
def extract_user_id(x):
  return x['user_id']

# Adapt the list of user ids to our user_ids_vocabulary object
user_ids_vocabulary.adapt(ratings.map(extract_user_id))

# Q5-2-0 Print the size of the user id vocabulary.
print(user_ids_vocabulary.vocabulary_size())

# Q5-2-1 Print the id of the 30th and 31st users (indices 29 and 30) from the user id vocabulary object.
print(user_ids_vocabulary.get_vocabulary()[29:31])

# 7. Build vocabulary to convert movie titles from the movies dataset into integer indices for embedding layers.
movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
# Adapt the object to the user_id feature of movie titles
movie_titles_vocabulary.adapt(movies)

# Q5-2-2 Print the size of the movie title vocabulary.
print(movie_titles_vocabulary.vocabulary_size())

# Q5-2-3 Print the id of the 30th and 31st titles(indices 29 and 30) from the movie title vocabulary object.
print(movie_titles_vocabulary.get_vocabulary()[29:31])

# 8. Define a Sequential user model using the user id vocabulary you built to create embedding with output dimension of 28.
user_model = tf.keras.Sequential([
  user_ids_vocabulary,
  tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 28)])

# Q5-2-4 Print the embeddings of user id 17
print(user_model(["17"]))

# Define a Sequential movie model using the movie title vocabulary you built to create embedding with output dimension of 28.
movie_model = tf.keras.Sequential([
  movie_titles_vocabulary,
  tf.keras.layers.Embedding(movie_titles_vocabulary.vocabulary_size(), 28)])

# Q5-2-5 Print the embeddings of Star Wars (1977)
print(movie_model(tf.constant(['Star Wars (1997)'])))

# 11. Make the task object batch movie features into batches of 256, and apply the movie model you built to each batch.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
  movies.batch(256).map(movie_model)))

# 12. Create a MovieLensModel object using the user model, movie model, and task you created above.
model = MovieLensModel(user_model, movie_model, task) # task from above in 11

# 13. Compile the model with the SGD optimizer with a .75 learning rate.
model.compile(optimizer=tf.keras.optimizers.SGD(0.75))

# 14. Train the model with the ratings dataset, batch size of 3500, and 3 epochs.
model.fit(ratings.batch(3500), epochs=3)

# 15. Create a streaming recommendation model using Tensorflow Recommenders (tfrs) library and the tfrs.layers.factorized_top_k.Streaming class.
index = tfrs.layers.factorized_top_k.Streaming(model.user_model)

# 16. Call the index_from_dataset method to build an index for the movies using a batch of 150 movies. 
# Use the map function to apply a lambda function to each title in the batch, which generates a tuple containing
# the movie title and the output of the movie model when it’s passed the title. The resulting dataset of title and movie embeddings is then used to build the index.
def map_function(title): # map function to apply 
  return (title, model.movie_model(title)) # tuple containing movie title as title, and output of the movie model when it's passed the title

# Call the index_from_dataset method to build an index for the movies using a batch of 150 movies
new_index = index.index_from_dataset(movies.batch(150).map(map_function))

# 17. Use the index created above to make top 5 recommendations for user 17
_, titles = new_index(np.array(["17"]))

# Q5-2-6 Print “Top 5 recommendations for user 17: xxx” Replace xxx with the recommended movie titles.
print(f'Top 5 recommendations for user 17: {titles[0, :5]}')
# Output at submission:
# Top 5 recommendations for user 17: [b'One Crazy Summer (1986)' b'Friday the 13th Part VI: Jason Lives (1986)'
# b'Trail of the Pink Panther (1982)'
# b'Do You Remember Dolly Bell? (Sjecas li se, Dolly Bell) (1981)'
# b'Bronson (2009)']
