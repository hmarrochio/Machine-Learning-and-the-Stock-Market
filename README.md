# CAPSTONE SPRINT 2 - README


### Random Matrix Theory and the Stock Market

Predicting the stock market is a very complicated task. I plan to use techniques from Physics and Mathematics, in particular insights from Random Matrix Theory, in order to characterize randomness in movements of stocks, and use it as regularization in order to explore
machine learning models of clustering, time series forecasting and incorporate these insights for portfolio strategies.



### Goal


Studying stock market data is a great laboratory for testing the efficacy of machine learning techniques. An important challenge in the industry is portfolio optimization: given risk and uncertainty, how can one maximize returns over time? Machine learning offers valuable insights here. For instance, unsupervised learning can be used to cluster together similar stocks, which is useful for portfolio diversification by avoiding focusing resources in similar stocks. In addition, the past few years have witnessed important developement in deep learning applications across industries, solving complex problems with non-linear predictive capabilities. Since some techniques are still quite recent, it is worth asking how effectively can deep learning techniques analyze historical stock data for improved predictions.


This project has two main objectives: first, to use real stock data as a learning tool for exploring and practicing various machine learning techniques; we actually investigate three different learning problems: unsupervised, supervised and reinforcement learning. Second, to investigate whether insights from Random Matrix Theory (RMT) can improve the predictions from these algorithms. While the practical application of these methods to actual investment strategies involves many nuances, this project focuses on exploring these ideas in a theoretical and experimental framework by analyzing historical data. 


By benchmarking the range of randomness in the data using RMT techniques, we aim to uncover the "true" signal between stocks. If the correlation between stocks are less influenced by random data, clustering algorithms should better identify stock similarities.  At least at this stage, this is exactly what we find: denoising the correlation matrix is a useful technique for clustering the correlation matrix! Among the three clustering algorithms tested, Agglomerative Clustering performed the best for this task.

Next, we shift gears to time series prediction. We analyze time series forecasting with RNN, GRU, and LSTM models, optimizing hyperparameters to predict Adj Close prices using Apple stock data. Training spans two years, with a four-month test set and a rolling window of 40 points. Models effectively capture price trends, generalizing well to other stocks like Walmart and Chevron. We also compare ARIMA, which achieves competitive MSE but seems to struggle with trend consistency, highlighting the strengths of neural networks for market predictions.

Finally, we trained a reinforcement learning algorithm to simulate trading using historical data. Given the high computational demand of RL training, we scaled down to a rolling window of size 3, with 2 months of training data and 1 month for testing. By carefully tweaking the reward function, we successfully mimicked a profitable trading strategy, demonstrating the potential of this approach even in a simplified model. Future work will focus on expanding this framework, incorporating longer time windows, exploring more complex reward structures, and improving trading constraints, such as adjusting the number of stocks that can be sold and bought each day.










### Data Description

The kaggle link for the dataset I will be investigating is https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks .

My data contains information about Standard and Poor (S&P-500), the top 500 companies listed trading in the United States. It contains 503 Symbol names because "Alphabet", "Fox Corporation" and "News Corporation" trade each under two different symbols. The data range from 2010 to present daily updates, but I consolidated by downloading it on October 22nd 2024. 

There are three csv files:

1) __sp500_companies.csv__

   This file contains 16 columns describing overall features relevant to the stock exchange of the companies featured in the S&P. The most important ones for our analysis are

   - `Symbol` : The symbol the stock is being exchanged as.
   - `Shortname`: The shortname of the company.
   - `Sector`: Sector that the company operates in.
   - `Weight`: Percentage of participation in the S&P market capital.
   
3) __sp500_index.csv__

   This file contains only 2 columns, `Date` and `S&P500`, which contains the `S&P500` index. At least for now, we will not be using this data.

5) __sp500_stocks.csv__

   This contains the main file we will do EDA and further model in this project. It contains 8 columns, but the relevant ones for the analysis are

   - `Date`: Date that the stock was traded. Notice that it is not every day of the year, since the stock market follows business days and observes certain holidays.
   - `Symbol`: The symbol the stock is being exchanged as.
   - `Adj Close`: The adjusted close price, it takes into account company actions, such as paying dividends as well as stock splits.


   We will also calculate the `Return` $\frac{P_{t} - P_{t-1}}{P_{t-1}}$ and `Log-return` $\mathrm{log} \left( \frac{P_{t}}{P_{t-1}}\right)$, where $P_t$ is the `Adj Close` price at day $t$.




### Overview about EDA (Notebook 1)

The first step of my EDA is simply to learn about the properties of the companies featured in S&P-500, in terms of participation in the stock market capital and which sectors are more represented. 

Another important aspect of the preliminary EDA is to understand the presence of null values. In our case, null values have meaning: if a stock only started being exchanged at a certain date - so in our example some time after 2010, it will __not have a price at earlier dates!__ There are a few options on how to deal with this fact. For the scope of this project, since $85.5$% of stocks have data in the full range of dates, we will make sure we only analyze stocks that do not have null values. This assumption would need to be revisited if we needed market information right now to make trades, but since we are looking at historical data and benchmarking models, we will choose the simpler assumption.

Next, we calculate return and log-return, selecting companies by 2 major criteria: top N stocks ordered by `Weight` or simply picking at random N stocks (always making sure there are no null values for the date range of interest). We can transform the data, choose a time period and then calculate correlation matrices between each stock.

At least for this part of the analysis, correlation matrices are the most important piece of data we construct. For such, we explore how to select the stocks to be analyzed, as well as the time period we use to calculate the correlation matrix with respect to `Return`. One difficulty is that using longer periods of time is useful for isolating signal, but it also dilutes many interactions between the stocks. Exploring how to navigate this tradeoff will be a major point during my modeling phase.

Finally, we can calculate the eigenvalues for the correlation matrices between stock symbols. By analyzing the structure of the distribution of the eigenvalues, we can isolate which range is most likely due to random correlations, by fitting to the expectation of RMT and the Marcenko-Pastur pdf.

### Eigenvalue Analysis and Denoising (Notebook 2-3)



First, let us introduce the expectation from RMT. If one is to sample correlation functions constructed by rectangular random matrices (size $T\times N$), in the large $T,N$ limit ($T,N \rightarrow  \infty$ with $T/N$ fixed), the statistics of the eigenvalues of the correlation matrix follows a specific pdf, called Marcenko-Pastur. This pdf depends on only two parameters,
- $\sigma$: Related to the variance of the components of the rectangular random matrix.
- $q$: Ratio between the dimensions of the random matrix, $T/N$. It is assumed that $q>1$.
  
One can verify this fact by generating enough data, as we show in the plot below.


![Eig](https://drive.google.com/uc?export=view&id=1Y0mi7VzlkdOFQBry2aX-qbhkqW05zxGY)

The important thing to notice is that the range of influence of __randomness is confined to a range of eigenvalues__! Therefore, if we can  identify which eigenvalues contribute to signal and randomness, the hope is that learning algorithms can make more precise predictions.


Next, we consider $100$ stocks and analyze a long period of data (5 years). One can see an isolation of the eigenvalues most likely due to randomness. We follow the procedure described in the book `Machine Learning for Asset Managers` by Lopéz de Prado, where we optimize the square error of the MP pdf to fit the data.


![Eig](https://drive.google.com/uc?export=view&id=1PkNn3fLebaBrvv4U4TyH9PRe_wWf-ihU)

We see here that most signal is within RMT range, but a few eigenvalues are clearly signal! A literature review shows this is pretty characteristic of stock data, where it is expected that most eigenvalues are in the random zone.

The RMT analysis has deep relation to Principal Component Analysis. In fact, PCA consists of precisely finding the directions in the data where the variance is largest, which is precisely done by evaluating the eigenvalue structure of correlation matrix! There are however two additional insights about characterizing the random range of eigenvalues:

1. The eigenvalues __outside__ the random range should correspond to signal, characterizing stocks that move together. We can therefore use this input as the number of relevant PCA directions.
2. Since the information present in the random range is most likely due to noise, it makes sense to __flatten__ the structure of the corresponding eigenvalues. This technique is called __denoising__.

Our qualitative analysis is that the clustering algorithm performed better after the denoising regularization. We follow the denoising procedure described in the Prado-Lopez book. In the image below, we show the log of the eigenvalues, before and after applying the denoising regularization.

![LogEig](https://drive.google.com/uc?export=view&id=130g0W28YxwgatYN8Kw1RSAlzrLXKNxqr)

Notice that for the eigenvalues __outside__ the random range, they remain the same. However, for the eigenvalues associated to randomness, denoising removes the structure to a flat plateau (while maintaining the trace of the correlation matrix). The expectation is that a clustering algorithm will more efficiently distinguish stocks as there are less structures due to randomness.



### Unsupervised Machine Learning Modeling - Clustering (Notebook 3)

For the first modeling analysis, we perform unsuperviser learning techniques in order to find stocks that are deeply related to each other. Notice that the model is agnostic in terms of industry: all the information we are analyzing is the `Log-Return` correlation between 100 stocks, using 5 years of data. 

Our main analysis consists of testing 3 different clustering algorithms with sklearn packages. We try `K-means`, `Agglomerative Clustering` and `Gaussian Mixture`, deciding on the size of clustering by the _sillhoutte score_. We also perform denoising as described in the previous section, as well as PCA for dimensional reduction focusing on the number of directions outside the random range.

Here we the _sillhoutte score_ with respect to the 3 algorithms analyzed. We choose `k=16` and `Agglomerative Clustering`, having fewer clustering classes could be useful to inspect qualitately if the clustering algorithm 

![Sil](https://drive.google.com/uc?export=view&id=1eb4IqxOv_99tp9F8d8XzwxwvYbWOGTUh)

Let us visually show what the clustering algorithm found in terms of the correlation matrix. First, this is the correlation matrix of the top 100 stocks, where the stocks are simply ordered alphabetically. Notice that there are some local structure but hard to pinpoint exactly which groups belong together.  


![UnClustHeat](https://drive.google.com/uc?export=view&id=1NFeP-Do0u-ZfQK2TdBzLGu1sAmdDMWCT)


In the figure below, we simply re-ordered the stocks with respect to the cluster labels produced by the learning algorithm! Simply by reordering the correlation matrix, we see lots of inter and intracluster structures of rectangular shape!

![ClustHeat](https://drive.google.com/uc?export=view&id=1VfSO8XKhZeq3aPLIyXiqa68XZ68EqPI9)

It is worth mentioning that by investigating the individual clusters, we can see which stocks are related (more details in the analysis of Jupyter Notebook 3). For instance, healthcare, financial and oil and gas stocks were each very clearly labelled in their own cluster group. We also found some interesting relations: big tech companies such as Microsfot and credit card companies such as Visa and Mastercard were in the same cluster group, which possibly is due to credit card companies and expandion of online shopping being so related.

This clustering analysis was done by denoising the correlation matrix. In Jupyter Notebook 3, we show that under similar circumstances, the clustering algorithm did not perform as well if the original data was not denoised. Despite similar sillhoutte score, the clusters generate a few classes with two many industries joined, as well as some classes almost empty - one of them only had `Google` stocks. So at least as a first impression, it seems that denoising was a useful technique for clustering.

### Superviser Learning - Forecasting Time Series - (Notebook 4)

In this section we present our time series forecasting analysis. We focus our efforts into investigating hyperparameters for a simple Recurrent Neural Network (RNN), as well as GRU and LSTM. The metric I am using both to evaluate model performance and training loss is `Mean Squared Error`. We divide the training into 2 years of trading data, the test as the next 4 months and rolling window of size $40$ data points.  We investigate `Adj Close` (price) time series for prediction. 

The summary of the best hyperparameters we found using the Apple stock is (we did not perform full time series cross validation, but we performed several experiments in Notebook 4 Appendices C-D):

|                       |  Simple RNN           | GRU        | LSTM       |
|:----------------------|:---------------------:|:----------:|:----------:|   
| __A__-Hidden States   |         40-80-40      |   40-80-40 |   40-80-40 |
| __B__ - Activation    |      Relu             |  Relu      |  Relu      |
| __C__ - Dropout       |      0.05             |      0.05  |     0.05   |
| __D__ - Batch         |                2      |      2     |    2       |
| __E__ - Learning Rate |               0.001   |     0.001  |    0.001   |
| __F__ - Optimizer     |              Adam     |     Adam   |  RMSprop   |
| __MSE train__         |           0.0029      |   0.0029   |  0.0035    |
| __MSE test__          |          0.0017       |   0.0011   |     0.0039 |


First, let us focus only on the test data range, for readability. We see the three models successfully captured the shape of the price trend.

![Price_focus](https://drive.google.com/uc?export=view&id=1n7rYT5OvBa8XXwwcLZGd5os9slMt6B05)

It is important to recall that since this is the test data, the model was __not trained__ having access to it, or in other words, the prices in this date range were not used as a target variable during training.

For completeness, we show here the full range of the neural network predictions for our data, both within training and test date range. 

![Price_all](https://drive.google.com/uc?export=view&id=1OguZr3hMkAcZWQcPtzmlD_6ujMeaUUKP)

Despite not performing a full time series cross validation, the architecture we fixed also successfully captured price trends for other stocks. We tested price variation from other industries, such that the movements are not necessarily directly correlated: Walmart, JP Morgan and Chevron. We show below the predictions for Walmart.

![Walmart](https://drive.google.com/uc?export=view&id=1uuiqK8habl1tVst904IIq_A5P1Bcf1wz)

Notice that despite not investigating the best hyperparameters for this particular stock, the models analyzed were successful in predicting the overall trends, as well as a reasonable MSE score. We summarize our findings below, more details in notebook 4.


|           | Simple RNN, MSE test        | GRU , MSE test        | LSTM, MSE test        |
|:----------|:---------------------------:|:---------------------:|:---------------------:|   
| Apple     |         0.0017              |         0.0011        |         0.0039        |
| Walmart   |         0.0004              |         0.0020        |         0.0012        |
| JP Morgan |         0.0013              |         0.0105        |         0.0016        |
| Chevron   |         0.0042              |         0.0022        |          0.0006       |

The last investigation in notebook 4 is related about alternative methods. We investigate ARIMA and a naive forecasting.

For ARIMA, we try to mimic the forecasting techniques of the Neural Network: given 40 feature points and a target (the next value in the time series), find a predictive model. We have to __fit a different model for every window__. The best ARIMA MSE was $0.0015$ for parameters (p=10,d=2,q=10). Despite being a smaller MSE than the Neural Network experiments, it seems that it oscillates more around the data than the NN models we investigated. Therefore, it might not have predicted the market trends as well the neural network ones. 


![ARIMA](https://drive.google.com/uc?export=view&id=1NRa3k4FezxHjLcc10gFUZYMwd-8ygKW_)

We did not experiment as much with the ARIMA hyperparameters. It would be interesting to investigate this further, as well as exploring different metrics to determine whether ARIMA or neural network predicted better the trends of the market.


### Reinforcement Learning Trader - (Notebook 5)

For the final Machine Learning model we explore, we change gears to reinforcement learning. The overall goal is to construct an agent (trader) that interacts with the environemnt and learns through iteractive rewards how to trade.

Training a very robust reinforcement learning of course is outside the scope of this project, my goal is at least to do a proof of concept and have the bases of a reinforcemen learning trader set up. We leave performing a detailed analysis for a future project.

The model analyzed is quite simple: There are three possible actions:

- 0: __Buy__
- 1: __Hold__
- 2: __Sell__

In addition, one restriction is that at each day, you can either buy or sell only one stock position. Once again, the goal here is not to construct a robust trader, but to start from a simple model that we can analyze on a personal computer as a laboratory for different incentives. 

The learning algorithm we consider is of a deep Q-learning problem. We do not train a deep neural network, but still choose this approach being a building block for 

There are a few steps for the model being analyzed:

__Pre-processing__

- A) Create rolling window of size 3 for 2 months of training data and 1 month of testing.
- B) Train a Neural Network to forecast the next price point in the rolling window. I will call this Neural Network __NN1__.
- C) Once NN1 is trained, we will __not__ update its weights.

__Q-Learning:__

- D) Start the Q-learning process. The state we will consider is the rolling window size of 3 prices (price today plus two previous ones), as well as the __forecasted price__ from NN1. So the state will consist of 4 price values.
- E) Train the network as usual, via a reward feedback, and a exploitation versus exploration trade-off.


It is interesting to describe a bit the logic behind a few of our reward options. 

__Incentives__:

- __Make small profit every day__

  $\text{portfolio}_\text{new} = \text{portfolio}_{\text{old}} + \left( \text{positions} \, \times \, \text{price}_{\text{today}} \right) $

  
  $\text{reward} = \frac{\text{portfolio}_\text{new}-\text{portfolio}_\text{old}}{\text{portfolio}_\text{old}} $
  
- __Compare prices with previous mean__:

  Suppose we have the average of the previous prices we have access to. Of course we can make a moving average over more sophisticated models, but for now we restrict ourselves to the previous points in the training data.

  We call two quantities the auxiliary mean and the auxiliary standard deviation

  $\mu_\text{aux} = \text{mean}(\text{prices} [\text{Day}1:\text{Today}])$

  $\sigma_\text{aux} =\text{std}(\text{prices} [\text{Day}1:\text{Today}])$

  We therefore add a reward that incentives selling when the price is above the auxiliary average, and buy when it is below. We divide by the standard deviation to account for market volatility.


  $\text{reward} = \frac{p-\mu_\text{aux}}{\sigma_\text{aux} }$



Notice how we were successful in increasing our `wealth`, despite how primitive the model is - for instance, we can't even sell or buy more than one position a day! The important trend is that whenever the price was increasing, we had an incentive to sell positions, adn when it was decreasing, we would hold or buy. Notice that the data to the right of the black line is unseen data. 


![Time](https://drive.google.com/uc?export=view&id=1ubs6dqB6dhhoLj3g47g-wYaZzMX4lNSn_)

This is the main plot of our Reinforcement Learning for now. Some very important properties: The RL trader was selling the positions as the price was high, then for a long time was alternating between buying and holding.

One very nice feature of this experiment is that whenever it saw another rise in price the test data, it started selling again to make a profit, and we see it was done selling by the last point, as comparing to the previous moving average, it is probably crossing a low price point again.


![Wealth](https://drive.google.com/uc?export=view&id=1ubs6dqB6dhhoLj3g47g-wYaZzMX4lNSn)




### Conclusion

This project had the main goal of applying and practicing different machine learning techniques to the stock market, which is a notoriously challenging problem to tackle. Making predictions in the stock market is difficult because of the mix of randomness and signal, so using multiple approaches and techniques is a good strategy to extract meaningful insights.

First, during the EDA, we pre-processed the stock time series and analyzed the properties of correlations between stocks. Using Random Matrix Theory, we were able to partly regularize the randomness from the signal in the correlation matrix. Denoising proved to be a useful tool for clustering algorithms, with Agglomerative Clustering performing the best. This helped us identify clear groups of stocks related by sector, such as healthcare, financials, or oil, as well as some (slightly) unexpected relationships, like big tech and credit card companies clustering together.

Next, for time series forecasting, we investigated RNN, GRU, and LSTM models, exploring their best hyperparameters to predict adjusted closing prices. These models successfully captured price trends for multiple stocks, including Apple, Walmart, and Chevron. When compared to ARIMA, neural networks seemed to perform better in terms of predicting market trends, even though ARIMA achieved similar MSE in some cases. This suggests that neural networks are better suited for non-linear time series forecasting.

Finally, we implemented a reinforcement learning trader as a proof of concept. The RL model was kept simple to allow for experimentation, with a rolling window of size 3 and basic trading actions (buy, hold, sell). By tweaking reward functions, we were able to simulate a successful trading strategy, showing the potential of RL in this space. Future work will focus on expanding the RL model, allowing for longer time windows, more complex reward functions, and greater flexibility in trading decisions, such as varying the number of stocks bought or sold each day.

Overall, this project shows how machine learning techniques can lead to interesting and meaningful results in stock market analysis. Random Matrix Theory was a useful tool for the clustering analysis, one possible investigation to the future is whether one can apply similar insights to forecasting and reinforcement learning. 









  