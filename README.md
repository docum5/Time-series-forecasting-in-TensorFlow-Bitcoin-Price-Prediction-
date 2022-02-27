# Time-Series-Prediction-with-TensorFlow
A series of different Time Series modellings experiments with various models to predict Bitcoin price



<p align="center">
  <img src="https://github.com/docum5/Time-series-forecasting-in-TensorFlow-Bitcoin-Price-Prediction-/blob/main/1584520546.jpeg?raw=true" />
</p>

## Table of Content
  * [The Problem](#the-problem)
  * [Goal](#goal)
  * [Project Main Steps](#project-main-steps)
  * [Modeling](#modeling)
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


Comparing the Performance of Each of Our Models           | Comparing the Performance by F1-score
:-------------------------:|:-------------------------:
![](https://github.com/docum5/Natural-Language-Processing-with-TensorFlow/blob/main/Screen%20Shot%202022-01-07%20at%2011.30.23.png?raw=true)  | ![](https://github.com/docum5/Natural-Language-Processing-with-TensorFlow/blob/main/Screen%20Shot%202022-01-07%20at%2011.29.53.png?raw=true)


## Conclusion
In this capstone project, I took a Kaggle challenge to classify tweets into disaster tweets in real or not?. First, I have analyzed and explored all the provided tweets data to visualize the statistical and other properties of the presented data. Next, I performed some exploratory analysis of the data to check the type of the data, whether there are unwanted features and if features have missing data. Based on the analysis, I decided to drop the "location" and "keyword" column since it has most of the data missing and really has no effect on the classification of tweets. The 'text' columns are all text data along with alphanumeric, special characters, and embedded URLs.The 'text' column data needs to be cleaned, pre-processed and vectorized before using a machine-learning algorithm to classify the tweets. After pre-processing the train and test data, the data was vectorized using CountVectorizer and TFIDF features. Then it was split into training and validation data, and then various classifiers were fit on the data, and predictions were made. Out of all classifiers tested, tf_hub_sentence_encoder(using pre-trained embedding universal sentence encoder) performed the best with the test accuracy of 81,1%. The second best choice model is Naive Bayes, with a test accuracy of 79,2%.


## Software and Libraries
This project uses the following software and Python libraries:



![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png" width=100>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/440px-NumPy_logo_2020.svg.png" width=150>](https://numpy.org/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1024px-Pandas_logo.svg.png" width=200>](https://pandas.pydata.org/docs/getting_started/index.html) [<img target="_blank" src="https://camo.githubusercontent.com/aeb4f612bd9b40d81c62fcbebd6db44a5d4344b8b962be0138817e18c9c06963/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f686f72697a6f6e74616c2e706e67" width=200>](https://www.tensorflow.org/) [<img target="_blank" src="https://matplotlib.org/stable/_static/logo2.svg" width=100 height=50>](https://matplotlib.org/)

