#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error


#Getting the stock data (install yfinance "pip install yfinance")
import yfinance as yf
ticker = yf.download('F', period="5y")
ticker.describe()

#adjusting data to what we need
tickerData = ticker
adjclose = tickerData['Adj Close']
tickerData['Today Adj Close'] = adjclose

tickerData['Adj Close'] = tickerData['Adj Close'].shift(-1)
tickerData = tickerData.dropna()
tickerData = tickerData.drop(['Close'], axis=1)
tickerData = tickerData.rename(columns={"Adj Close" : "Tomorrow Adj Close"})
#tickerData = tickerData.drop(['Volume'], axis=1)

attributes = ['Open', 'High', 'Low', 'Tomorrow Adj Close', 'Today Adj Close']
#literally don't use this at all
scale = ['Open', 'High', 'Low', 'Today Adj Close']
scaler = MinMaxScaler()
#feature_transform = scaler.fit_transform(tickerData[scale])
#feature_transform= pd.DataFrame(columns=attributes, data=feature_transform, index=tickerData.index)


#Getting the data for training

#get the labels seperated from the features
train_set, test_set = train_test_split(tickerData, test_size=0.2, random_state=42)
train_set = train_set.drop(['Volume'], axis=1)
test_set = test_set.drop(['Volume'], axis=1)

train_set_features = train_set.drop(['Tomorrow Adj Close'], axis=1)
train_set_labels= train_set["Tomorrow Adj Close"].copy()
feature_transform = scaler.fit_transform(train_set_features[scale])
train_prepared = pd.DataFrame(columns=scale, data=feature_transform, index=train_set_features.index)

test_set_features = train_set.drop(['Tomorrow Adj Close'], axis=1)

test_set_labels= train_set["Tomorrow Adj Close"].copy()

#linear Regression Model to see how it does
svm_reg = LinearSVR(epsilon=.5)
svm_reg.fit(train_prepared, train_set_labels)

pred = svm_reg.predict(test_set_features)
                                
#rmse with lin_reg
scores = cross_val_score(svm_reg, train_prepared, train_set_labels, 
                         scoring="neg_mean_squared_error", cv =10) 

rmse = np.sqrt(-scores)
print("Scores:", rmse)
print("Mean:", rmse.mean())
print ("Standared deviation:", rmse.std())
print("MAPE: ", mean_absolute_percentage_error(pred, train_set_labels))


print(train_set_labels.describe())                                   


# In[22]:


from ta import add_all_ta_features

tickerData = add_all_ta_features(
    tickerData, open="Open", high="High", low="Low", close="Tomorrow Adj Close", volume="Volume", fillna=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# In[30]:


train_set['Predicted Close'] = pred.tolist()
tickerData['Today Adj Close'].plot(label = 'Actual Close', figsize = (50,24))
#train_set['Predicted Close'].plot(label = 'Predicted Close')
plt.legend

