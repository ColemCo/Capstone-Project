#!/usr/bin/env python
# coding: utf-8

# In[47]:


from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[67]:


#Getting the stock data (install yfinance "pip install yfinance")
import yfinance as yf
ticker = yf.download('AAPL', period="5y")
test_length = 31 #days of data taken out of training for testing
ticker.describe()


# In[65]:


#Saving last month of data to test with
testingData = ticker.tail(test_length)


# In[70]:


#Getting the data for training (not including last month)
trainingData = ticker.head(len(ticker)-test_length)
temp = trainingData.reset_index()
trainingData = temp

adjclose = trainingData['Adj Close']
trainingData['Today Adj Close'] = adjclose

trainingData['Adj Close'] = trainingData['Adj Close'].shift(-1)
trainingData = trainingData.dropna()
trainingData = trainingData.drop(['Close'], axis=1)
trainingData = trainingData.rename(columns={"Adj Close" : "Tomorrow Adj Close"})
trainingData


# In[72]:


from pandas.plotting import scatter_matrix

attributes = ['Open', 'High', 'Low', 'Tomorrow Adj Close', 'Volume', 'Today Adj Close']

scatter_matrix(trainingData[attributes], figsize=(12,8))
corr_matrix = trainingData.corr()
corr_matrix['Tomorrow Adj Close']


# In[24]:


#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(ticker[attributes])
feature_transform= pd.DataFrame(columns=attributes, data=feature_transform, index=ticker.index)
feature_transform.head()


# In[25]:


from pandas.plotting import scatter_matrix

attributes = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

scatter_matrix(feature_transform[attributes], figsize=(12,8))
corr_matrix = feature_transform.corr()
corr_matrix['Adj Close']


# In[ ]:





# In[ ]:





# In[ ]:




