'''
M5 Lab-2: Recommender
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

from typing import Dict, Text
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
# FREEZE CODE END

'''
Load the movielens/100k-ratings train dataset
Load the movielens/100k-movies train dataset
Select movie title and user id from the ratings dataset
Select movie title from the movies dataset
'''

# Load the movielens/100k-ratings train dataset
ratings = tfds.load('movielens/100k-ratings', split='train') # users
# Load the movielens/100k-movies train dataset
movies = tfds.load('movielens/100k-movies', split='train') # movies

# Select movie title and user id from the ratings dataset
def transform_ratings(x):
  return { # return a dictionary with two keys - movie title and user id from the ratings dataset
    'movie_title': x['movie_title'],
    'user_id': x['user_id']
  }

# Select movie title from the movies dataset
def transform_movies(x):
  return x['movie_title'] # returns the movie title

ratings = ratings.map(transform_ratings)
movies = movies.map(transform_movies)


'''
Use the MovieLendsModel class provided here to define your own user and movie models
'''

# FREEZE CODE BEGIN
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
# FREEZE CODE END

'''
Build vocabulary to convert user ids into integer indices for embedding layers
Build vocabulary to convert movie titles into integer indices for embedding layers
Define a Sequential user model using the user id vocabulary you built to create embedding with output dimension of 50
Define a Sequential movie model using the movie title vocabulary you built to create embedding with output dimension of 50
Define a TFRS Retrieval task object with FactorizedTopK as the metric
Make the task object batch  movie features into batches of 128, and apply the movie model you built to each batch
**Q5-2-0 Print the embeddings of user id 5
'''

# Build vocabulary to convert user ids into integer indices for embedding layers
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
#  create a user-defined function to extract the user id
def extract_user_id(x):
  return x['user_id']
# adapt the list of user ids to our user_ids_vocabulary object
user_ids_vocabulary.adapt(ratings.map(extract_user_id))

# Build a vocabulary to convert movie titles to integer indices for embedding layers
movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
# Adapt the object to the user_id feature of the movie titles
movie_titles_vocabulary.adapt(movies)

# Define a Sequential user model using the user id vocabulary you built to create embedding with output dimension of 50
user_model = tf.keras.Sequential([
  user_ids_vocabulary,
  tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 50)
])

# Define a Sequential movie model using the movie title vocabulary you built to create embedding with output dimension of 50
movie_model = tf.keras.Sequential([
  movie_titles_vocabulary,
  tf.keras.layers.Embedding(movie_titles_vocabulary.vocabulary_size(), 50)
])

# Define a TFRS Retrieval task object with FactorizedTopK as the metric
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
# Make the task object batch movie features into batches of 128, and apply the movie model you built to each batch
  movies.batch(128).map(movie_model)
))

# **Q5-2-0 Print the embeddings of user id 5
print(user_model(["5"]))

'''
Create a MovieLensModel object using the user model, movie model, and task you created above
Compile the model with the Adagrad optimizer with a .65 learning rate
Train the model with the ratings dataset, batch size of 5000, and 2 epochs
'''

# Create a MovieLensModel object using the user model, movie model, and task you created above
model = MovieLensModel(user_model, movie_model, task)
# user_model = embeddings
# movie_model = embeddings
# task = Retrieval object from above

# Compile
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.65))

# Train the model with the ratings dataset, batch size of 5000, and 2 epochs
model.fit(ratings.batch(5000), epochs=2)

'''
Use this code snippet to create a brute-force nearest neighbor search index for a recommendation model.
'''

# FREEZE CODE BEGIN
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
def map_function(title):
    return (title, model.movie_model(title))

new_index = index.index_from_dataset(movies.batch(100).map(map_function))
# FREEZE CODE END

'''
Use the index created above to make top 5 recommendations for user 23
**Q5-2-1 Print "Top 5 recommendations for user 23
'''
# get recommendations for user 23
# _, titles = new_index(np.array(["23"]))
# print(f'Top 5 recommendations for user 23: {titles[0, :5]}')
print(f'Top 5 recommendations for user 23')
