import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math  
import sklearn.metrics  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
data = pd.read_csv('./Data/hour.csv')
data.head()
data.columns
data.rename(columns = {'instant':'index', 'dteday':'datetime',
'yr':'year', 'mnth':'month', 'holiday':'is_holiday',
'workingday':'is_workingday','weathersit':'weather_conditions','hum':'humidity',
'hr':'hour', 'cnt':'count'}, inplace = True)
data.drop(['index','casual','registered'], axis = 1, inplace = True)

data['datetime'] = pd.to_datetime(data.datetime)
# categorical variables
data['season'] = data.season.astype('category')
data['is_holiday'] = data.is_holiday.astype('category')
data['weekday'] = data.weekday.astype('category')
data['weather_conditions'] = data.weather_conditions.astype('category')
data['is_workingday'] = data.is_workingday.astype('category')
data['month'] = data.month.astype('category')
data['year'] = data.year.astype('category')
data['hour'] = data.hour.astype('category')


plt.figure(figsize = (8,6))
sns.heatmap(data.corr(),annot = True, cmap = 'BrBG')
data.drop(['atemp'], inplace = True, axis = 1)
data.head()
data.drop(['datetime'], axis = 1, inplace = True)
Y = data['count']
X = data.drop('count', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, pred)
rmse = math.sqrt(mse)
print(rmse)

model_2 = RandomForestRegressor(n_estimators = 200, max_depth = 15)
model_2.fit(X_train, Y_train)
pred = model_2.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, pred)
rmse = math.sqrt(mse)
print(rmse)

pipeline = Pipeline(steps = [
('model',model_2)
])

model = pipeline.fit(X_train, Y_train)
predictions = pipeline.predict(X_test)
print(math.sqrt(sklearn.metrics.mean_squared_error(Y_test, predictions)))