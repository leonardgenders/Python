# CHALLENGE #1: Working with two csv files to write a MapReduce program which calculates the avg rating for each movie. 
# Need to round the average rating to one decimal place and the output should include the movie name and 
# avg rating for each movie.
----------------------------
from mrjob.job import MRJob
from mrjob.step import MRStep

class avgMovieRating(MRJob):

  # for adding in the movieNames.csv file
  def configure_args(self):
    super(avgMovieRating, self).configure_args()
    self.add_file_arg('--items', help='movieRatings.csv')

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetRatings,
      reducer_init=self.reducer_init,
      reducer=self.reduceravgRatings)
    ]
  
  def mapperGetRatings(self, _, line):
    (userID, movieID, rating, timestamp) = line.split(',')
    yield movieID, float(rating) # originally had 'movieID, 1' I think this is where the error in the output being 1 as the avg rating (fixed)

# to replace movieID with moviename
  def reducer_init(self):
    self.movieNames = {}

    with open("movieNames.csv", encoding='ascii', errors='ignore') as f:
      for line in f:
        fields=line.split(',')
        self.movieNames[fields[0]] = fields[1]
  
  # combing into one reducer instead of trying two seperate
  # original output without the for loop is moviename, sum ratings
  def reduceravgRatings(self, key, values):
    count_ratings = 0
    sum_ratings = 0
    for v in values:
      count_ratings += 1
      sum_ratings += v
    rating_avg = round((sum_ratings/count_ratings), 1)
    yield self.movieNames[key], rating_avg
    
if __name__ == '__main__':
  avgMovieRating.run()


----------------------------
# CHALLENGE #2: There may be an issue with some movies only being watched by a few people and avg ratings are therefore
# not as reliable. Write a MapReduce program that calculates the avg rating for every movie with at least 200 ratings. 
# Round the avg rating to one decimal place and the output should include the movie name, the total number of ratings and
# the avg rating for each movie.
----------------------------
from mrjob.job import MRJob
from mrjob.step import MRStep

class avgMovieRating200(MRJob):

  # add in the movieNames.csv file
  def configure_args(self):
    super(avgMovieRating200, self).configure_args()
    self.add_file_arg('--items', help='movieRatings.csv')

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetRatings,
      reducer_init=self.reducer_init,
      reducer=self.reducerAvgRatings200)
    ]

  def mapperGetRatings(self, _, line):
    (userID, movieID, rating, timestamp) = line.split(',')
    yield movieID, float(rating) # need two?
  
  # to replace movieID with moviename
  def reducer_init(self):
    self.movieNames = {}

    with open("movieNames.csv", encoding='ascii', errors='ignore') as f:
      for line in f:
        fields=line.split(',')
        self.movieNames[fields[0]] = fields[1]
  
  def reducerAvgRatings200(self, key, values):
    count_ratings = 0
    sum_ratings = 0
    for v in values:
      count_ratings += 1 # this is a counter that counts the number of actual ratings (rating value does not matter here)
      sum_ratings += v # this sums the actual rating values
    if (count_ratings) >= 200:
      rating_avg = round((sum_ratings/count_ratings), 1)
      yield self.movieNames[key], (count_ratings, rating_avg) # originally had sum_ratings but that was summing the rating values
      # only need the sum of the ratings count to ensure over 200

if __name__ == '__main__':
  avgMovieRating200.run()


----------------------------
# CHALLENGE #3: Write a MapReduce program that calculates the avg rating for every movie with at least 200 ratings. 
# Round the avg rating to one decimal place and the output should include the movie name, the total number of ratings and
# the avg rating for each movie. Sort the output by total number of ratings from smallest to largest. 
----------------------------


from mrjob.job import MRJob
from mrjob.step import MRStep

class sortedAvgMovieRating200(MRJob):

  # add in the movieNames.csv file
  def configure_args(self):
    super(sortedAvgMovieRating200, self).configure_args()
    self.add_file_arg('--items', help='movieRatings.csv')

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetRatings,
      reducer_init=self.reducer_init,
      reducer=self.reducerAvgRatings200),
      MRStep(mapper=self.mapperPrepForSort,
      reducer=self.reducerSort)
    ]

  def mapperGetRatings(self, _, line):
    (userID, movieID, rating, timestamp) = line.split(',')
    yield movieID, float(rating)
  
  # to replace movieID with moviename
  def reducer_init(self):
    self.movieNames = {}

    with open("movieNames.csv", encoding='ascii', errors='ignore') as f:
      for line in f:
        fields=line.split(',')
        self.movieNames[fields[0]] = fields[1]

  def reducerAvgRatings200(self, key, values):
    count_ratings = 0
    sum_ratings = 0
    for v in values:
      count_ratings += 1
      sum_ratings += v
    if (count_ratings) >= 200:
      rating_avg = round((sum_ratings/count_ratings), 1)
      yield self.movieNames[key], (count_ratings, rating_avg)

  def mapperPrepForSort(self, movienames, rating_info):
    count_ratings, rating_avg = rating_info
    yield None, (count_ratings, rating_avg, movienames) # need to include movienames here for reducer output below

  def reducerSort(self, key, value):
    sort_output = sorted(value) # sorts by the first value, so placed 'count_ratings' above in [0] position
    for v in sort_output:
      yield v[2], (v[0], v[1]) # output will be movienames, then tuple of (count_ratings, rating_avg)

if __name__ == '__main__':
  sortedAvgMovieRating200.run()
