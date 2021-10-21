#!/usr/bin/env python
# coding: utf-8

# In[80]:


from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[81]:


#Getting the stock data (install yfinance "pip install yfinance")
import yfinance as yf
ticker = yf.download('SPY', period="ytd")
test_length = 31 #days of data saved from training for testing
ticker


# In[82]:


#Saving last month of data to test with
testingData = ticker.tail(test_length)
testingData


# In[83]:


#Getting the data for training (not including last month)
trainingData = ticker.head(len(ticker)-test_length)
temp = trainingData.reset_index()
trainingData = temp


# In[84]:


#Makes empty lists to store days and adj close prices
days = list()
adj_close_prices = list()


# In[86]:


#Gets the date and Adj close column individually and stores separately 
train_days = trainingData.loc[:, 'Date']
train_adj_close = trainingData.loc[:, 'Adj Close']


# In[91]:


#creates list of days for the training data.
for day in train_days:
    days.append(day)
    
#creates list of closing prices for the training data.
for adj_close_price in train_adj_close:
    adj_close_prices.append(float(adj_close_price))


# In[92]:


print(adj_close_prices)


# In[95]:


print(days)


# In[ ]:




