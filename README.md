# Demand-Data-Forecast

# CODE FOR DEMAND FORECASTING OF CEMENT

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 156)
pd.set_option('display.max_columns',20)

# Importing dataset

dataset1 = pd.read_excel("Book2.xlsx", sheet_name='Sheet1')
dataset = dataset1.copy()

# Plotting Dataset to check for stationarity, trend, seasonality

A = dataset["Demand"]
B = dataset["Month"]
plt.figure(figsize=(10, 6))
plt.plot(B, A, marker='o', linestyle='-', color='b', label='Tracking Signal')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.title('Demand vs. Time')
plt.legend()
plt.grid()
plt.show()

print("We can see that the Demand Data is having Seasonality as well as Trend so I am applying Winter's Method to Forecast the Demand")

# Deseasonalized Data

dataset['MA'] = np.nan
for i in range(dataset.shape[0]-24):
    idx = list(range(i + 1, i + 12))
    if i + 13 < dataset.shape[0]:  # Check if the index is within bounds
        dataset.loc[i + 6, "MA"] = (dataset.loc[i, 'Demand'] + dataset.loc[i + 12, 'Demand'] +2* np.sum(dataset.loc[idx, 'Demand']))/24
    else:
        dataset.loc[i + 6, "MA"] = (dataset.loc[i, 'Demand'] +2* np.sum(dataset.loc[idx, 'Demand']))/24


# Finding Lo and To for the deseasonalised data using Linear Regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X = np.array(range(7,139)).reshape(-1, 1)
y = np.array(dataset.loc[range(6, 138), 'Demand']).reshape(-1, 1)
regressor.fit(X, y)

# Initializing some variables for Forecasting using Winter's Method

Lo = regressor.intercept_[0]
To = regressor.coef_[0][0]
alpha = 0.05
beta = 0.1
gamma = 0.1

# Finding the Level, Trend and Seasonal Factors for the Demand Data

dataset["Deseaonalized Demand"] = Lo + To*dataset['No.']
dataset["Seasonal Factor"] = dataset["Demand"]/dataset["Deseaonalized Demand"]

for i in range(12):
    idx = range(i,dataset.shape[0],12)
    dataset.loc[i, 'SF'] = np.average(dataset.loc[idx, 'Seasonal Factor'])

for i in range(dataset.shape[0]):
    if i==0:
        dataset.loc[i, "Level"] = alpha*(dataset.loc[i, "Demand"]/dataset.loc[i, "SF"]) + (1-alpha)*(Lo+To)
        dataset.loc[i, "Trend"] = beta*(dataset.loc[i, "Level"] - Lo) + (1-beta)*(To)
    elif i<=11:
        dataset.loc[i, "Level"] = alpha*(dataset.loc[i, "Demand"]/dataset.loc[i, "SF"]) + (1-alpha)*(dataset.loc[i-1, "Level"] + dataset.loc[i-1, "Trend"])
        dataset.loc[i, "Trend"] = beta*(dataset.loc[i, "Level"] - dataset.loc[i-1, "Level"]) + (1-beta)*(dataset.loc[i-1, "Trend"])
    if i>11:
        dataset.loc[i, "Level"] = alpha*(dataset.loc[i, "Demand"]/dataset.loc[i-12, "SF"]) + (1-alpha)*(dataset.loc[i-1, "Level"] + dataset.loc[i-1, "Trend"])
        dataset.loc[i, "Trend"] = beta*(dataset.loc[i, "Level"] - dataset.loc[i-1, "Level"]) + (1-beta)*(dataset.loc[i-1, "Trend"])
        dataset.loc[i, "SF"] = gamma*(dataset.loc[i, "Demand"]/dataset.loc[i, "Level"]) + (1-gamma)*(dataset.loc[i-12, "SF"])
        
# Forecast the Demand and finding Deviation, Absolute Deviation

dataset.loc[0, "Forecast"] = (Lo + To)*dataset.loc[0, "SF"]
for i in range(1, dataset.shape[0]):
    dataset.loc[i, "Forecast"] = (dataset.loc[i-1, "Level"] + dataset.loc[i-1, "Trend"])*dataset.loc[i, "SF"]
    
dataset["Deviation"] = dataset["Forecast"] - dataset["Demand"]
dataset["Absolute Deviation"] = np.abs(dataset["Deviation"])

# Compute MAD, MSE, Deviation %, MAPE and Tracking Signal

for i in range(dataset.shape[0]):
    dataset.loc[i, "MADi"] = (dataset.loc[:i, "Absolute Deviation"]).sum()/(i+1)
    dataset.loc[i, "MSEi"] = (dataset.loc[:i, "Deviation"]**2).sum()/(i+1)
    dataset.loc[i, "Deviation %"] = (dataset.loc[i, "Absolute Deviation"]/dataset.loc[i, "Demand"])*100
    dataset.loc[i, "MAPEi"] = dataset.loc[i ,"Deviation %"].sum()/(i+1)
    dataset.loc[i, "Tracking Signal i"] = (dataset.loc[:i, "Deviation"].sum())/dataset.loc[i, "MADi"]
    
# Plotting the Tracking Signal Vs. Time Chart to monitor the accuracy and performance of the forecast model

A = dataset["Tracking Signal i"]
B = dataset["Month"]
plt.figure(figsize=(10, 6))
plt.plot(B, A, marker='o', linestyle='-', color='b', label='Tracking Signal')
plt.xlabel('Time')
plt.ylabel('Tracking Signal')
plt.title('Tracking Signal vs. Time')
plt.legend()
plt.grid()
plt.show()

# Plotting Control Chart of X-bar of the Forecast Error/Deviation

forecast_errors = dataset["Deviation"]
mean_error = np.mean(forecast_errors)
std_deviation = np.std(forecast_errors)

# Set control limits (mean +/- 2 times standard deviation)

upper_control_limit = mean_error + 2 * std_deviation
lower_control_limit = mean_error - 2 * std_deviation

plt.figure(figsize=(10, 6))
plt.plot(dataset["Month"], forecast_errors, marker='o', linestyle='-', color='b', label='Forecast Errors')
plt.axhline(y=mean_error, color='r', linestyle='--', label='Mean Error')
plt.axhline(y=upper_control_limit, color='g', linestyle=':', label='Upper Control Limit')
plt.axhline(y=lower_control_limit, color='g', linestyle=':', label='Lower Control Limit')
plt.xlabel('Time')
plt.ylabel('Forecast Error')
plt.title('X-bar Control Chart for Forecasting Demand')
plt.legend()
plt.grid()
plt.show()

# Final Dataset with all the computed values

print(dataset)
