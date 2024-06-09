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


# Challenge 7: Write a Spark program that:
# Create an RDD named accountsRDD that captures the data from the accounts.csv file.
# Create an RDD named zipCodesRDD that captures the data from the us_zip_codes.csv file.
# Select first name, last name, and zip code from accountsRDD and store them in filteredAccRDD.
# Select zip code, city, and state from zipCodesRDD and store them in filteredZipCodesRDD.
# Create a data frame named accountsDF based on filteredAccRDD. Make sure the column names follow the ones in the original data set.
# Create a data frame named zipCodesDF based on filteredZipCodesRDD. Make sure the column names follow the ones in the original data set.
# Create a new data frame from accountsDF with just the first_name and last_name columns and only include users whose last name begins with the letter B. Display the first 10 records.
# Create a new data frame that joins accountsDF and zipCodesDF by the zip code. The columns should be in the following order: first_name, last_name, city, state, zip, and the rows should be sorted based on last name. Display the first 10 records.
# Calculate the number of accounts in each city and sort them by the count in a descending order. Display the first 10 records.

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

def mapperAccounts(line):
  fields = line.split(',')
  return(str(fields[0]), str(fields[1]), int(fields[4]))

def mapperZips(line_zip):
  fields_zip = line_zip.split(',')
  return(int(fields_zip[0]), str(fields_zip[1]), str(fields_zip[2]))

# Create an RDD named accountsRDD that captures the data from accounts.csv
accountsRDD = sc.textFile("accounts.csv")
# Select first name, last name, and zip code from accountsRDD and store them in filteredAccRDD
filteredAccRDD = accountsRDD.map(mapperAccounts)

# Create an RDD named zipCodesRDD that captures the data from the us_zip_codes.csv file
allzipCodesRDD = sc.textFile("us_zip_codes.csv")
# Remove the header
header = allzipCodesRDD.first()
zipCodesRDD = allzipCodesRDD.filter(lambda line: line != header)
# Select zip code, city, and state from zipCodesRDD and store them in filteredZipCodesRDD
filteredZipCodesRDD = zipCodesRDD.map(mapperZips)

# Create a df named accountsDF based on filteredAccRDD
accSchema = StructType([StructField("first_name", StringType(), True), \
                        StructField("last_name", StringType(), True), \
                        StructField("zip", IntegerType(), True)])
accountsDF = spark.createDataFrame(filteredAccRDD, accSchema)
accountsDF.show(5)


# Create a df named zipCodesDF based on filteredZipCodesRDD
zipSchema = StructType([StructField("zip", IntegerType(), True), \
                        StructField("city", StringType(), True), \
                        StructField("state", StringType(), True)])
zipCodesDF = spark.createDataFrame(filteredZipCodesRDD, zipSchema)
zipCodesDF.show(5)

# Create a new df from accountsDF with just the first_name and last_name cols
# only include users whose last name begins with the letter B, display first 10 records
# using SQL regular expression for those that start with capital B, drop the zip col, show first 10
accountsDF.where("last_name REGEXP '^B'").drop('zip').show(10)

# Create a new df that joins accountsDF and zipCodesDF by the zip code. The cols should be in the following 
# order: first_name, last_name, city, state, zip, and the rows should be sorted based on last name.
# Display the first 10 records
new_joinDF = accountsDF.join(zipCodesDF, "zip", "left_outer")
# split out seperately after the join and filter out NAs from city?
new_joinDF.select('first_name', 'last_name', 'city', 'state', 'zip').where(new_joinDF.city.isNotNull()).orderBy('last_name').show(10)


# Calculate the number of accounts in each city and sort them by the count in a descending order.
# Display the first 10 records.
new_joinDF.filter(new_joinDF.city.isNotNull()).groupBy('city').count().sort('count', ascending=False).show(10)


# Challenge 8: Write a Spark program that conducts the following:
# Find all flights whose distance is greater than 600 miles and had an early departure of at least 10 minutes. Display the first 10 records.
# Find all flights between San Francisco (SFO) and Chicago (ORD) with at least a two-hour delay. Sort the data frame by the delay in descending order. Display the first 10 records.

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# load the departuredelays.csv file into a df
departureDF = spark.read.format("csv").option("header", "true").load("departuredelays.csv")
departureDF.show(10)

# find all flights whose distance is greater than 600 miles and had an early departure
# of at least 10 minutes. Display the first 10 records.
departureDF.createOrReplaceTempView("departureDFview1")
miles600 = spark.sql("SELECT * FROM departureDFview1 WHERE distance > 600 AND delay <= -10")
# select the cols in order as required by example output
miles600.select('distance', 'delay', 'origin', 'destination').show(10)


# Find all flights between San Francisco (SFO) and Chicago(ORD) with at least a two-hour delay
# Sort the df by the delay in descending order. Display the first 10 records.
departureDF.createOrReplaceTempView("departureDFview2")
# need delay col as integer, not sorting numerically
sfoToOrd = spark.sql("SELECT *, CAST(delay AS INT) as int_delay FROM departureDFview2 WHERE origin == 'SFO' AND destination == 'ORD' AND delay >= 120")
sfoToOrd.select('date', 'int_delay', 'origin', 'destination').withColumnRenamed("int_delay", "delay").sort('delay', ascending=False).show(10)


# Chellenge 9: We would like to make our analysis more user-friendly. Please create a new data frame by implementing the following:
# Label all flights, regardless of origin and destination, with an indication of the delays they experienced. Use the labels Very Long Delays for delays greater than 6 hours, Long Delays for delays between 2 to 6 hours, Short Delays for delays between 1 to 2 hours and Tolerable Delays for delays less than 1 hour. Since there are flights with no delays or early departure use the labels No Delays and Early for those, respectively. Add these user friendly labels in a new column called Flight_Delays.
# Instead of airport codes for the origin, use airport names.
# Question:
# Find the number of flights from each origin, which have delays that passengers cannot tolerate. Sort the data frame by the number of non-tolerable delays in a descending order. Display the first 10 records.

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

