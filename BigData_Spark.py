# CHALLENGE 1: Write a Spark program to find the minimum lowest stock price for each stock.
# Output should include the stock_ticker first and then price, sorted in descending
# order from large to small. Remove the header row.

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("MinPrice")
sc = SparkContext(conf = conf)

allPricesRDD = sc.textFile("stock_info.csv")
header = allPricesRDD.first()
pricesRDD = allPricesRDD.filter(lambda line: line != header)

pricesRDDnew = pricesRDD.map(lambda line: line.split(','))

stockPricePair = pricesRDDnew.map(lambda x: (x[0], float(x[4])))

minPrices = stockPricePair.reduceByKey(lambda x, y: min(x,y))
minPricesSortedByStock = minPrices.sortBy(lambda x: x[1], False)

for row in minPricesSortedByStock.collect():
  print(row)


# CHALLENGE 2: Write a Spark program to find the total number of COVID-19
# cases in each state. Final RDD needs to include only states with total
# number of cases greater than 100,000 and should be sorted in ascending order.
# Remove any headers as necessary.

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("MinPrice")
sc = SparkContext(conf = conf)

allPricesRDD = sc.textFile("stock_info.csv")
header = allPricesRDD.first()
pricesRDD = allPricesRDD.filter(lambda line: line != header)

pricesRDDnew = pricesRDD.map(lambda line: line.split(','))

stockPricePair = pricesRDDnew.map(lambda x: (x[0], float(x[4])))

minPrices = stockPricePair.reduceByKey(lambda x, y: min(x,y))
minPricesSortedByStock = minPrices.sortBy(lambda x: x[1], False)

for row in minPricesSortedByStock.collect():
  print(row)


# Challenge 3: Develop a Spark program to determine the total amount each customer spends.
# Round the amount to a single decimal and sort in ascending order. 

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("TotalSpending")
sc = SparkContext(conf = conf)

# read in the file and remove the header
allCustomersRDD = sc.textFile("customerinfo.csv")
header = allCustomersRDD.first()
customersRDD = allCustomersRDD.filter(lambda line: line != header)

# csv, so split by comma
customersRDDnew = customersRDD.map(lambda line: line.split(','))

# pair of custID and amountSpent should be the two cols, float amountSpent
custAmountPair = customersRDDnew.map(lambda x: (x[0], float(x[2])))

amountSum = custAmountPair.reduceByKey(lambda x, y: x+y)

amountRounded = amountSum.mapValues(lambda x: round(x,1))

# sort least to most, so use True
custCountSorted = amountRounded.sortBy(lambda x: x[1], True)

for row in custCountSorted.collect():
  print(row)


# Challenge 4: Develop a Spark program to calculate word frequency
# in the file and only keep words that have 2 or more occurences.

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("WordCountOver1")
sc = SparkContext(conf = conf)

lines = sc.textFile("posting.txt")

# split by spaces because txt file not csv, split by spaces not commas
words = lines.flatMap(lambda x: x.split())

# counter for each word
wordOne = words.map(lambda x: (x,1))

wordCount = wordOne.reduceByKey(lambda x,y: x+y)

# add in filter for words that occurred more than once (>1)
wordsMoreThanOne = wordCount.filter(lambda x: x[1] > 1)

wordCountSorted = wordsMoreThanOne.sortBy(lambda x: x[1], False)

for row in wordCountSorted.collect():
  print(row)


# Challenge 5: Develop a Spark program using SparkSQL and data frames to calculate the average movie rating
# Round the rating to 4 decimals, sort the df by movieID and display the first 12 observations.

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# load the movieRatings.csv file into a df
movieRatingsDF = spark.read.format("csv").option("header", "false").load("movieRatings.csv")

# rename cols
movieRatingsDF = movieRatingsDF.select("_c0", "_c1", "_c2", "_c3").withColumnRenamed("_c0", "userID") \
                                                                  .withColumnRenamed("_c1", "movieID") \
                                                                  .withColumnRenamed("_c2", "rating") \
                                                                  .withColumnRenamed("_c3", "timestamp")

movieRatingsDF.createOrReplaceTempView("movieRatingsDFview")
# cast movieID as integer
# round the sum of the ratings divided by the count of ratings to four decimals, group by movieid
avgs = spark.sql("SELECT CAST(movieID as INT) as MovieID, ROUND((SUM(rating)/COUNT(rating)),4) AS AverageRating FROM movieRatingsDFview GROUP BY movieID")
# order by movieID and in ascending order, show the first 15 observations
avgs.select('MovieID', 'AverageRating').orderBy('MovieID', ascending = True).show(12)



# Challenge 6: Create a Spark program using SparkSQL and data frames to find the mean movie rating
# to 4 decimals and only show movies with atleast 200 ratings, output sort descending for avg ratings
# Use col names ("MovieName", "AverageRating", "NumberOfRatings")

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# load the movieNames.csv file into a df
namesDF = spark.read.format("csv").option("header", "false").load("movieNames.csv")
# rename cols in namesDF
namesDF = namesDF.select("_c0", "_c1", "_c2").withColumnRenamed("_c0", "movieID").withColumnRenamed("_c1", "MovieName")\
                                                                                  .withColumnRenamed("_c2", "genre")
namesDF.show(5)

# load the movieRatings.csv file into a df
ratingsDF = spark.read.format("csv").option("header", "false").load("movieRatings.csv")
# rename cols in ratingsDF
ratingsDF = ratingsDF.select("_c0", "_c1", "_c2", "_c3").withColumnRenamed("_c0", "userID") \
                                                        .withColumnRenamed("_c1", "movieID") \
                                                        .withColumnRenamed("_c2", "rating") \
                                                        .withColumnRenamed("_c3", "timestamp")
ratingsDF.show(5)


# join names and ratings on movieID
new_joinDF = ratingsDF.join(namesDF, "movieID", "left_outer")
new_joinDF.createOrReplaceTempView("ratingsview")

# create the avg ratings by taking the sum of the ratings / the count of the ratings and naming as AverageRating
# create the total number of ratings by taking the count of the ratings and naming as NumberOfRatings
# group by the MovieName (which is possible bc of join above), WHERE cannot do aggregate functions so need to use
# HAVING COUNT instead of WHERE COUNT, can use alias 'AverageRating' for ORDER BY and complete in descending order (large to small)
movieRatingInfoDF = spark.sql("SELECT MovieName, ROUND((SUM(rating)/COUNT(rating)), 4) AS AverageRating, COUNT(rating) AS NumberOfRatings \
FROM ratingsview GROUP BY MovieName HAVING COUNT (rating) >= 200 ORDER BY AverageRating DESC").show()
