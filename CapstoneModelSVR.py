#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Getting the stock data (install yfinance "pip install yfinance")
import yfinance as yf
ticker = yf.download('AAPL', period="5y")
ticker.describe()

#adjusting data to what we need
tickerData = ticker
adjclose = tickerData['Adj Close']
tickerData['Today Adj Close'] = adjclose

tickerData['Adj Close'] = tickerData['Adj Close'].shift(-1)
tickerData = tickerData.dropna()
tickerData = tickerData.drop(['Close'], axis=1)
tickerData = tickerData.rename(columns={"Adj Close" : "Tomorrow Adj Close"})
tickerData = tickerData.drop(['Volume'], axis=1)

attributes = ['Open', 'High', 'Low', 'Tomorrow Adj Close', 'Today Adj Close']
#literally don't use this at all
scale = ['Open', 'High', 'Low', 'Today Adj Close']
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(tickerData[attributes])
feature_transform= pd.DataFrame(columns=attributes, data=feature_transform, index=tickerData.index)



#Getting the data for training

#get the labels seperated from the features
train_set, test_set = train_test_split(feature_transform, test_size=0.2, random_state=42)
train_set_features = train_set.drop(['Tomorrow Adj Close'], axis=1)
train_set_labels= train_set["Tomorrow Adj Close"].copy()
                            
test_set_features = train_set.drop(['Tomorrow Adj Close'], axis=1)
test_set_labels= train_set["Tomorrow Adj Close"].copy()
   
#linear Regression Model to see how it does
svm_reg = LinearSVR(epsilon=.5)
svm_reg.fit(train_set_features, train_set_labels)
                                   
#print("Predictions:" , lin_reg.predict(test_set_features))
#print("Labels:", list(test_set_labels))

#rmse with lin_reg
stock_predictions = svm_reg.predict(test_set_features) 
mse = mean_squared_error(test_set_labels, stock_predictions)
rmse = np.sqrt(mse)
print(rmse)                                   
print(tickerData.describe())


# In[7]:


#adds predictions to the stock dataframe in order to print on graph
train_set['Predicted Close'] = stock_predictions.tolist()

feature_transform['Tomorrow Adj Close'].plot(label = 'Actual Close', figsize = (50,24))
train_set['Predicted Close'].plot(label = 'Predicted Close')
plt.legend()


# In[3]:




