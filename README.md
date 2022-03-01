# Time-Series-Prediction-with-TensorFlow
A series of different Time Series modelings experiments with various models to predict Bitcoin price



<p align="center">
  <img src="https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/1584520546.jpeg?raw=true" />
</p>

## Table of Content
  * [The Problem](#the-problem)
  * [Goal](#goal)
  * [Project Main Steps](#project-main-steps)
  * [Data Visualization](#data-visualization)
  * [Modeling](#modeling)
  * [Prediction](#prediction)
  * [Conclusion](#conclusion)
  * [Software and Libraries](#software-and-libraries)


## The Problem

Python Bitcoin is widely used cryptocurrency for digital market. It is decentralised that means it is not own by government or any other company.Transactions are simple and easy as it doesn’t belong to any country.Records data are stored in Blockchain.Bitcoin price is variable and it is widely used so it is important to predict the price of it for making any investment.

## Goal
This project focuses on the accurate prediction of cryptocurrencies price using tensorflow. 

## Project Main Steps:

* Get time series data (the historical price of Bitcoin)
* Format data for a time series problem
  * Creating training and test sets 
  * Visualizing time series data
  * Turning time series data into a supervised learning problem (windowing)
  * Preparing univariate and multivariate (more than one variable) data
* Evaluating a time series forecasting model
* Setting up a series of deep learning modelling experiments
  * Dense (fully-connected) networks
  * Sequence models (LSTM and 1D CNN)
  * Ensembling (combining multiple models together)
  * Multivariate models
  * Replicating the N-BEATS algorithm using TensorFlow layer subclassing
* Creating a modelling checkpoint to save the best performing model during training
* Making predictions (forecasts) with a time series model
* Creating prediction intervals for time series model forecasts

## Data Visualization

Price of Bitcoin from 4 November 2014 to 2 Feb 2022           | Price of Bitcoin from 4 November 2014 to 2 Feb 2022 Train&Test 
:-------------------------:|:-------------------------:
![](https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/visualisasi1.png)  | ![](https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/visualisasi2.png)


## Modeling

```
Data -> Format data for a time series problem -> build a model -> Evaluating -> Making predictions (forecasts)
```


| Model                                                                 | mae         | mse         | rmse        | mape     | mase     |
|-----------------------------------------------------------------------|-------------|-------------|-------------|----------|----------|
| naive_model                                                           | 1127.009888 | 2704483.50  | 1644.531494 | 2.823276 | 0.998678 |
| model_1_dense_w7_h1                                                   | 1140.201172 | 2764825.75  | 1662.776489 | 2.847246 | 1.008707 |
| model_2_dense_w30_h1                                                  | 1233.389771 | 3079271.75  | 1754.785522 | 3.085911 | 1.084139 |
| model_3_dense_w30_h7                                                  | 2419.106689 | 11996798.00 | 2770.479492 | 6.062185 | 2.125433 |
| model_4_CONV1D                                                        | 1144.036255 | 2792920.00  | 1671.203125 | 2.851182 | 1.012100 |
| model_5_LSTM                                                          | 1187.783813 | 2923671.50  | 1709.874756 | 2.962411 | 1.050802 |
| model_6_multivariate                                                  | 1139.666504 | 2739723.25  | 1655.210938 | 2.848582 | 1.008234 |
| model_8_NBEATs                                                        | 1180.154175 | 2880262.50  | 1697.133789 | 2.962500 | 1.044052 |
| model_9_ensemble                                                      | 1157.829346 | 2824341.75  | 1680.577759 | 2.891772 | 1.024302 |


Comparing the Performance by mae of Each of Our Models          
:-------------------------:
![](https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/maeconclusion.png) 

## Prediction

Prediction of bitcoin price with interval values         | Price of Bitcoin Prediction to Future 
:-------------------------:|:-------------------------:
![](https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/forecast%20w:%20interval.png?raw=true)  | ![](https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/prediction%20to%20future.png?raw=true)


## Conclusion
In this capstone project, I took Bitcoin prices from [CoinDesk](https://www.coindesk.com/price/bitcoin/), analyzed them, applied various models to fit the data, and forecasted the models. The predictions we have made here are not financial advice. Furthermore, by now, we should be well aware of just how poor machine learning models can be at forecasting values in an open system - anyone promising you a model which can "beat the market" is likely trying to scam you, oblivious to their errors or very lucky.


## Software and Libraries
This project uses the following software and Python libraries:



![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/440px-NumPy_logo_2020.svg.png" width=150>](https://numpy.org/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1024px-Pandas_logo.svg.png" width=200>](https://pandas.pydata.org/docs/getting_started/index.html) [<img target="_blank" src="https://camo.githubusercontent.com/aeb4f612bd9b40d81c62fcbebd6db44a5d4344b8b962be0138817e18c9c06963/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f686f72697a6f6e74616c2e706e67" width=200>](https://www.tensorflow.org/) [<img target="_blank" src="https://matplotlib.org/stable/_static/logo2.svg" width=100 height=50>](https://matplotlib.org/)

