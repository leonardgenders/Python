'''
M5 Assignment-1: Build an embedding model
'''
# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import pprint

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# Import additional lirbaries/modules as needed
# FREEZE CODE END

# 1. Load the training set of the movielens/latest-small-movies dataset from Tensorflow datasets
movies = tfds.load('movielens/latest-small-movies', split='train')

# 2. Q5-1-0: Print the first 2 records of the train set using the pprint function
for x in movies.take(2).as_numpy_iterator():
  pprint.pprint(x)

# 3. Use a Tensorflow/Keras StringLookup() layer object to build a vocabulary of the movie_title feature
movie_title_lookup = tf.keras.layers.StringLookup()
movie_title_lookup.adapt(movies.map(lambda x: x['movie_title']))

# 4. Q5-1-1 Print the size of the vocabulary you just built
print(movie_title_lookup.vocabulary_size())

# 5. Q5-1-2 Print the titles of the 30th and 31st movies (indices 29 and 30) from the vocabulary object you just built. They should look like this:
# ['Titanic1 (1997)', 'Titanic2 (1998)']
print(movie_title_lookup.get_vocabulary()[29:31])

# 6. Q5-1-3 Print the raw tokens (i.e. movie titles) of the two movies from above as embedding ids (i.e. integer ids). They should look like this:
# tf.Tensor([[1][2]], shape=(2, 1), dtype=int64)
print(movie_title_lookup(["Zodiac (2007)", "Zipper (2015)"]))

# 7. Define a keras Embedding object with vocabulary size as the input dimension, and 28 as the output dimension
movie_title_embedding = tf.keras.layers.Embedding(
  input_dim = movie_title_lookup.vocabulary_size(),
  output_dim = 28)

# 8. Put the two together into a single layer which takes raw text in and yields embeddings.
movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])

# 9. Q5-1-4 Print the embeddings of Star Wars (1977)
print(movie_title_model(tf.constant(['Star Wars (1977)'])))
