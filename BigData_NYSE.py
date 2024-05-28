# CHALLENGE #1: NYSE.csv file includes stock data from the New York Stock Exchange. Each line includes eight values separated by a comma. 
# Write a MapReduce program that calculates the average adjusted closing price (stock_price_adj_close) for each stock.
-----------------------------
from mrjob.job import MRJob
from mrjob.step import MRStep

class avgStockPrice(MRJob):

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetPrices,
      reducer=self.reducerGetPriceAvg)]

# mapper that takes input of each column and returns the stock_symbol col and stock_price_adj_close as a float
# so that it can be iterable
  def mapperGetPrices(self, _, line):
    (stock_symbol, date, stock_price_open, stock_price_high, 
    stock_price_low, stock_price_close, stock_volume, 
    stock_price_adj_close) = line.split(',')
    yield stock_symbol, float(stock_price_adj_close)

# reducer which counts the adjusted closing price by yielding the stock_symbol, then a tuple with the 
# sum of all the stock_price_adj_close for each stock_symbol and count of the number of entries for each
# stock as count_stock_symbols
  def reducerGetPriceAvg(self, stock_symbol, values):
    count_stock_symbols = 0 # making a counter for instances of each stock stock_symbol
    sum_prices = 0 # making a counter for the sums
    for v in values:
      sum_prices += v # take the sum and add v which is each individual entry for the stock_price_adj_close
      count_stock_symbols += 1 # counter adds +1 for each instance of the stock_symbol
    price_avg = round((sum_prices/count_stock_symbols), 2) # all in one reducer vs multiple like before
    yield stock_symbol, price_avg
  
if __name__ == '__main__':
  avgStockPrice.run()


-----------------------------
# CHALLENGE #2: NYSE.csv file includes stock data from the New York Stock Exchange. Each line includes eight values separated by a comma. 
# Write a MapReduce program that calculates the average adjusted closing price (stock_price_adj_close) for each stock with a total volume of at least 100,000,000.
# The output should include stock ticker, total volume and average adjusted closing price for each stock. 
-----------------------------
# import the necessary packages
from mrjob.job import MRJob
from mrjob.step import MRStep

class avgStockPrice100M(MRJob):

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetPrices,
      reducer=self.reducerGetPriceAvg100MVol)]

  def mapperGetPrices(self, _, line):
    (stock_symbol, date, stock_price_open, stock_price_high, stock_price_low, stock_price_close, stock_volume,
    stock_price_adj_close) = line.split(',')
    yield stock_symbol, (int(stock_volume), float(stock_price_adj_close)) #****when you have more than two items to yield,
    # you need to put them in a tuple****

  def reducerGetPriceAvg100MVol(self, stock_symbol, stock_info): # revise to note above as stock_info
   # stock_info contains the volume and the stock_price_adj_close 
    count_stock_symbols = 0
    sum_prices = 0
    sum_vol = 0
    for vol, v in stock_info: # from above
      count_stock_symbols += 1
      sum_prices += v
      sum_vol += vol
    if (sum_vol) >= 100000000:
      price_avg = round((sum_prices/count_stock_symbols), 1)
      yield stock_symbol, (price_avg, sum_vol)

if __name__ == '__main__':
  avgStockPrice100M.run()


----------------------------
# CHALLENGE #3: NYSE.csv file includes stock data from the New York Stock Exchange. Each line includes eight values separated by a comma. 
# Write a MapReduce program that calculates the average adjusted closing price (stock_price_adj_close) for each stock with a total volume of at least 100,000,000.
# The output should include stock ticker, total volume and average adjusted closing price for each stock. Sort the output so that it includes total volume from smallest to largest.
-----------------------------
from mrjob.job import MRJob
from mrjob.step import MRStep

class sortedAvgStockPrice100M(MRJob):

  def steps(self):
    return[
      MRStep(mapper=self.mapperGetPrices,
      reducer=self.reducerGetPriceAvg100MVol),
      MRStep(mapper=self.mapperPrepForSort,
      reducer=self.reducerSort)
    ]

  def mapperGetPrices(self, _, line):
    (stock_symbol, date, stock_price_open, stock_price_high, stock_price_low,
    stock_price_close, stock_volume, stock_price_adj_close) = line.split(',')
    yield stock_symbol, (int(stock_volume), float(stock_price_adj_close)) # remember, 
    # when you have more than two items to yield you need to put them in a tuple

  def reducerGetPriceAvg100MVol(self, stock_symbol, stock_info):
    # stock_info contains the vol and stock_price_adj_close
    count_stock_symbols = 0
    sum_prices = 0
    sum_vol = 0
    for vol, v in stock_info: #from above
      count_stock_symbols += 1
      sum_prices += v
      sum_vol += vol
    if (sum_vol) >= 100000000:
      price_avg = round((sum_prices/count_stock_symbols), 1) # 1 decimal in output
      yield stock_symbol, ('%10d'%sum_vol, price_avg)
  
  def mapperPrepForSort(self, stock_symbol, stock_information):
    sum_vol, price_avg = stock_information
    yield None, (sum_vol, stock_symbol, price_avg)

  def reducerSort(self, key, value):
    sort_output = sorted(value)
    for v in sort_output:
      yield v[1], (v[0], v[2])

if __name__ == '__main__':
  sortedAvgStockPrice100M.run()
