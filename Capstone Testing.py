#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Getting the stock data (install yfinance "pip install yfinance")
import yfinance as yf
ticker = yf.download('AAPL', period="ytd")
test_length = 31 #days of data taken out of training for testing
ticker.describe()


# In[3]:


#Saving last month of data to test with
testingData = ticker.tail(test_length)
testingData


# In[16]:


#Getting the data for training (not including last month)
trainingData = ticker.head(len(ticker)-test_length)
temp = trainingData.reset_index()
trainingData = temp
trainingData

trainingData['Adj Close'] = trainingData['Adj Close'].shift(-1)
trainingData = trainingData.dropna()
trainingData = trainingData.drop(['Close'], axis=1)
trainingData


# In[18]:


#Makes empty lists to store days and adj close prices
days = list()
adj_close_prices = list()


# In[19]:


#Gets the date and Adj close column individually and stores separately 
train_days = trainingData.loc[:, 'Date']
print(train_days)
train_adj_close = trainingData.loc[:, 'Adj Close']
print(train_adj_close)


# In[20]:


#creates list of days for the training data.
for day in train_days:
    days.append(day)
    
#creates list of closing prices for the training data.
for adj_close_price in train_adj_close:
    adj_close_prices.append(float(adj_close_price))


# In[24]:


from pandas.plotting import scatter_matrix

attributes = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

scatter_matrix(ticker[attributes], figsize=(12,8))
corr_matrix = trainingData.corr()
corr_matrix['Adj Close']


# In[25]:


#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(ticker[attributes])
feature_transform= pd.DataFrame(columns=attributes, data=feature_transform, index=ticker.index)
feature_transform.head()


# In[26]:


from pandas.plotting import scatter_matrix

attributes = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

scatter_matrix(feature_transform[attributes], figsize=(12,8))
corr_matrix = feature_transform.corr()
corr_matrix['Adj Close']


# In[ ]:




