# Description:
# 1. The Optimisation of stock portfolio is something that plays a very significant role while investing in stocks.
# 2. By optimising stock portfolio, we can maximise our annual returns and earn good profit.
# 3. The following program attempts to optimise the stock portfolio of the user using Efficiet Frontier.
# 4. We will be applying the concept of maximising Sharpe Ratio to do so.

#Importing python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Get the stock symbols/tickers in the portfolio
# Here, we will take a finctional portfolio of FAANG.

# FAANG is an acronym referring to 5 best performing technology related American stocks, namely

# Facebook
# Amazon
# Apple
# Netflix
# Google

assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

# Now, let's assign weights to the stocks.
# Initially, all the stocks will have same weights which should add up to 1.

# All the stocks having same weights initially signifies that we are not being partial to a particular stock.

# So, the weights for the each stock here will be 1/5 = 0.2

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Now, get the stock portfolio starting date.
startingDate = '2015-01-01'

# Now, get the stock portfolio ending date which is today
today = datetime.today().strftime('%Y-%m-%d')
# print("Today is: ", today)
# print("")

# Now, we will create a dataframe to store the adjusted close price of the stocks chosen for our portfolio since starting date till today. We will use Yahoo Finance for the data we need.

#Create a dataframe to store the adjusted close price of stocks
df = pd.DataFrame()

#Store the adjusted close price of the stocks in df
for stock in assets:
  df[stock] = web.DataReader(stock, data_source = 'yahoo', start = startingDate, end = today)['Adj Close']

#Let's take a look at the dataframe
# print(df.head())
# print("")

# Let's visualise our stock portfolio by plotting the graph
for c in df.columns.values:
  plt.plot(df[c], label = c)

plt.title('Portfolio Adjusted Close Price History')
plt.xlabel('Date')
plt.ylabel('Adj. Close Price in USD')
plt.legend(df.columns.values, loc = 'upper left')
plt.show()

# Now, let's take a look at the daily returns of the stocks
returns = df.pct_change()
# print(returns.head())

# Now, let's create an annualised covariance matrix.
annual_cov_matrix = returns.cov() * 252
# print(annual_cov_matrix)

# Now, calculate the portfolio variance
port_var = np.dot(weights.T, np.dot(annual_cov_matrix, weights))
# print(port_var)

# Calculation of portfolio volatility or standard deviation
port_vol = np.sqrt(port_var);
# print(port_vol)

# Calculate the annual portfolio return
annual_port_return = np.sum(returns.mean() * weights) * 252
# print(annual_port_return)

# Let's take a look at the expected returns, volatility(risk) and variance.
percent_var = str( round(port_var, 3) * 100) + '%'
percent_vol = str( round(port_vol, 3) * 100) + '%'
percent_return = str( round(annual_port_return, 3) * 100) + '%'

print("Expected Annual return: " + percent_return)
print("Expected Volatility: " + percent_vol)
print("Expected Variance: " + percent_var)

# Observation:
# Looking at the above values, we can say that our portfolio has performed well over the time period chosen with the annual returns of nearly 35% with risk nearly 25%.
# But I think we can do better in terms of annual returns. So, let's dive in again to look what we can do to get better results.

#import the libraries
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Portfolio Optimisation!

#Calculate the expected returns and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
#mu stands for mean in mathematics

S = risk_models.sample_cov(df)

# print(mu)
# print(S)

# Now, Optimise for the Maximum Sharpe Ratio
# Sharpe Ratio basically tells us about how much access returns can we get for some extra amount of volatility/risk.

# It basically measures the performance of a risky investment and compares it to the performance of a risk-free investment.

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# print(cleaned_weights)

print(ef.portfolio_performance(verbose = True))

# Observation:
# The results above show a clear and astonishing development.

# By increasing the risk by 10% of what it actually was, we get 25% more annual returns of what they actually were.

# So, we can say that now we have got a more optimised portfolio.

# Let's take a look that how much percentage of our money should we invest in each stock

# Convert cleaned_weights (type : Collections.OrderDict) into a list
items = list(cleaned_weights.items())

print("We should invest following percentage of money in following stocks: ")
for stock in range(0,5):
  print(items[stock][0], "-> ", items[stock][1] * 100, "%")

# Conclusion:
# By taking less risk and being impartial towards every stock in our portfolio, we will get a good annual return.

# But, by optimising the portfolio using Sharpe Ratio, we can see that taking a little bit of more risk can give us better annual return.

# We also come to know about what percentage of our money should we invest in a particular stock to get those better results.