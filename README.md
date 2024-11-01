# CAPSTONE SPRINT 1 - README


### Random Matrix Theory and the Stock Market

Predicting the stock market is a very complicated task. I plan to use techniques from Physics and Mathematics, in particular insights from Random Matrix Theory, in order to characterize randomness in the dataset, and use it as a regularization in order to explore
machine learning models of clustering and (if time permits) forecasting.



### Goal

My goal is to investigate whether characterizing randomness from RMT techniques can help learning algorithms for better prediction in the stock market.

The expectation is that by benchmarking the range of randomness in the data, we can highlight the "true" signal between stocks. I will then investigate clustering algorithms, possibly K-clustering with distance metrics constructed with respect to the correlation matrix, as well as other information theory quantities, such as mutual information.

If time permits, I want also to investigate simple forecasting models that take into account the results from clustering in order to distribute the portfolio weigths accross different stocks. One interesting possiblity is to construct a simple model with reinforcement learning.


### Data Description

https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

My data contains information about Standard and Poor (S&P-500), the top 500 companies listed trading in the United States. It contains 503 Symbol names because "Alphabet", "Fox Corporation" and "News Corporation" trade each under two different symbols. It contains data from 2010 and is updated daily. I consolidated by downloading it on October 22nd 2024. 

There are three csv files:

1) __sp500_companies.csv__

   This file contains 16 columns describing overall features relevant to the stock exchange of the companies featured in the S&P. The most important ones for our analysis are

   - `Symbol` : The symbol the stock is being exchanged as.
   - `Shortname`: The shortname of the company
   - `Sector`: Sector that the company operates
   - `Weight`: Percentage of participation in the S&P market capital.
   
3) __sp500_index.csv__

   This file contains only 2 columns, `Date` and `S&P500`, which contains the `S&P500` index. At least for now, we will not be using this data.

5) __sp500_stocks.csv__

   This contains the main file we will do EDA and further model in this project. It contains 8 columns, but the relevant ones for the analysis are

   - `Date`: Date that the stock was traded. Notice that it is not every day of the year, since the stock market follows business days and observes certain holidays.
   - `Symbol`: The symbol the stock is being exchanged as.
   - `Adj Close`: The adjusted close price, it takes into account company actions, such as paying dividends as well as stock splits.


   We will also calculate with `Adj Close` the `Return` $\frac{P_{t} - P_{t-1}}{P_{t-1}}$ and `Log-return` $\mathrm{log} \left( \frac{P_{t}}{P_{t-1}}\right)$, where $P_t$ is the `Adj Close` price at day $t$.




### Overview about EDA

The first step on my EDA is simply to learn about the properties of the companies in S&P-500, which ones are the companies that has a bigger participation in market capital, which sectors are more represented. 

Another important aspect of the preliminary EDA is to understand the presence of null values. In our case, null values have meaning: if a stock only started being exchanged at a certain date - so in our example some time after 2010, it will __not have a price at earlier dates!__ There are a few options on how to deal with this fact. For the scope of this project, since $85.5$% of stocks do not have null values in the time interval represented by the data, we will make sure we only analyze stocks that do not have null values. This assumption would need to be revisited if we needed market information right now to make trades, but since we are looking at historical data, and we have enough of it, we will choose the simpler assumption.

Next, we calculate return and log-return, selecting companies by 2 major criteria: top N stocks ordered by `Weight` or simply picking at random N stocks (always making sure there are no null values for the date range of interest). We can transform the data, choose a time period and then calculate correlation matrices between each stock.

At least for this part of the analysis, correlation matrices are the most important piece of data we construct. For such, we explore how to select the stocks to be analyzed, as well as the time period we use to calculate the correlation matrix with respect to `Return`. One difficulty is that using longer periods of time is useful for isolating signal, but it also dilutes many interactions between the stocks. Exploring how to navigate these constraints will be a major contender during my capstone.

Finally, we can calculate the eigenvalues for the correlation matrices between stock symbols. By analyzing the structure of the distribution of the eigenvalues, we can isolate which range is most likely due to random correlations, by fitting to the expectation of RMT and the Marcenko-Pastur pdf.

### Preliminary Results


For now, I am describing the main result from EDA so far.


First, let us introduce the expectation from RMT. If one is to sample correlation functions constructed by rectangular random matrices (size $T\times N$), in the large $T,N$ limit ($T,N \rightarrow  \infty$ with $T/N$ fixed), the statistics of the eigenvalues of the correlation matrix follows a specific pdf, called Marcenko-Pastur. For instance, one can verify this fact by generating enough data, as we show in the plot below.


![Eig](https://drive.google.com/uc?export=view&id=1Vniufo2MudVenKXDcMmgCCbw_XR8OM34)

The important thing to notice is that the range of influence of __randomness is confined to a range of eigenvalues__! Therefore, if we can  identify our data between signal and randomness, the hope is that learning algorithms can make more precise predictions.


Now, we consider $100$ stocks and analyze a long period of data (5 years). One can see an isolation of the eigenvalues most likely due to randomness. We follow the procedure described in the book `Machine Learning for Asset Managers` by Lopéz de Prado. 


![Eig](https://drive.google.com/uc?export=view&id=1PkNn3fLebaBrvv4U4TyH9PRe_wWf-ihU)

We see here that most signal is within RMT range, but a few eigenvalues are clearly signal 

The next step is to use a regularization scheme for the noisy eigenvalues, called __"denoising"__ and investigate clusters within the stocks.