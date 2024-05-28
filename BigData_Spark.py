# CHALLENGE 1: Write a Spark program to find the minimum lowest stock price for each stock.
# Output should include the stock_ticker first and then price, sorted in descending
# order from large to small. Remove the header row.

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("MinPrice")
sc = SparkContext(conf = conf)

allPricesRDD = sc.textFile("NYSE_H.csv")
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

allPricesRDD = sc.textFile("NYSE_H.csv")
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
allCustomersRDD = sc.textFile("custSpendingH.csv")
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

lines = sc.textFile("blog.txt")

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
