# CAPSTONE SPRINT ! README


### Random Matrix Theory and the stock Market

Predicting the stock market is a very complicated task. I plan to use techniques from Physics and Mathematics, in particular insights from Random Matrix Theory, in order to characterize randomness in the dataset, and use it as a regularization in order to explore
clustering and forecasting.





### Describe your goal

My goal is to investigate whether characterizing randomness from RMT techniques can help learning algorithms for better prediction in the stock market.

The expectation is that benchmarking the range of randomness in the data, we can strengthen the "true" correlation between stocks. I will then investigate clustering algorithms, possibly investigate K-clustering with distance metrics constructed with respect to the correlation matrix, as well as other information theory quantities, such as mutual information.

If time permits, I want also to investigate simple forecasting models that take into account the results from clustering in order to distribute the portfolio weigths accross different stocks. One interesting possiblity is to construct a simple model with reinforcement learning.


### Describe your data

https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

My data contains information about Standard and Poor (S&P-500), list of the top 500 companies trading in the stock market. It contains data from 2010 and is updated daily, I consolidated by downloading it on October 22nd 2024. 

There are three csv files:

1) __sp500_companies.csv__

   This file contains 16 columns describing overall features relevant to the stock exchange of the companies in S&P. The most important ones for us are

   - `Symbol` : The symbol the stock is being exchanged at.
   - `Shortname`: The shortname of the company
   - `Sector`: Sector that the company operates
   - `Weight`: Percentage of participation in the S&P market capital.
   
3) __sp500_index.csv__

   fefefefe

5) __sp500_stocks.csv__

   fefefefe



The adjusted price, by the data description, takes into account company actions, such as paying dividends as well as stock splits. It should lead to a more consistent long term analysis of the stock price.



### Describe your work (models, analysis, EDA, etc.)



### Describe your results

