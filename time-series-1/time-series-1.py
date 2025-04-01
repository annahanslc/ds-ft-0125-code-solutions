# %% [markdown]
# # Time Series 1 Assignment

# %%
# imports

import pandas as pd
import matplotlib as plt
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# %%
# import data

air = pd.read_csv('AirPassengers.csv')
air.head()

# %%
# convert the Month to datetime

air['Month'] = pd.to_datetime(air['Month'])
air.info()

# %%
# set the Month to be the index

air = air.set_index('Month')
air.head()

# %%
# manually set the frequency

air.index.freq = 'MS'
air.info()

# %% [markdown]
# ### 1. Plot the time series data with rolling mean and rolling standard deviation and see if it is stationary.

# %%
# calculate the rolling mean

air['rolling_mean12'] = air['#Passengers'].rolling(12).mean()

# %%
# calculate the rolling standard deviation

air['rolling_std12'] = air['#Passengers'].rolling(12).std()

# %%
# check the dataframe with the rolling results

air.head(15)

# %%
# plot the dataframe to analyze rolling data

air.plot()

# %% [markdown]
# - There is systematic upward movement in the rolling mean over time. The beginning mean and ending mean are noticeably different. This means that the data does not have a constant mean. The data contains a trend.
# - The variance, plotted using the rolling standard deviation, shows an increasing trend in the variance. This means that the variance is not constant.
# - Because the data shows trend and has non-constant variance, it is NOT stationary.

# %% [markdown]
# ### 2. Try different levels of differences, and plot the time series data with rolling mean and standard deviation. See if it is stationary.

# %%
# trying 1 level of difference
# make a copy of the dataframe for stationary data
air_diff = air.copy()

air_diff['#Passengers'] = air_diff['#Passengers'].diff()
air_diff['rolling_mean12'] = air_diff['#Passengers'].rolling(12).mean()
air_diff['rolling_std12'] = air_diff['#Passengers'].rolling(12).std()
air_diff.plot()

# %% [markdown]
# - the rolling mean is constant after taking 1 level of diff
# - the rolling mean starts and end at the same level

# %%
# trying 2 levels of differences
# make a copy of the dataframe for stationary data
air_diff_diff = air.copy()

air_diff_diff['#Passengers'] = air_diff_diff['#Passengers'].diff().diff()
air_diff_diff['rolling_mean12'] = air_diff_diff['#Passengers'].rolling(12).mean()
air_diff_diff['rolling_std12'] = air_diff_diff['#Passengers'].rolling(12).std()
air_diff_diff.plot()

# %% [markdown]
# - the rolling mean becomes even more constant when taking 2 levels of diff

# %%
# trying 3 levels of differences
# make a copy of the dataframe for stationary data
air_diff_diff_diff = air.copy()

air_diff_diff_diff['#Passengers'] = air_diff_diff_diff['#Passengers'].diff().diff().diff()
air_diff_diff_diff['rolling_mean12'] = air_diff_diff_diff['#Passengers'].rolling(12).mean()
air_diff_diff_diff['rolling_std12'] = air_diff_diff_diff['#Passengers'].rolling(12).std()
air_diff_diff_diff.plot()

# %% [markdown]
# - At 3 levels of diff, there does is not significant improvement from 2 levels of diff

# %% [markdown]
# ### 3. Try to transform the data, and make different levels of differences. See if it is stationary.

# %% [markdown]
# - sqrt, cube root, log

# %%
# use log to transform the data
# copy the dataframe
air_log = air.copy()

air_log['#Passengers'] = np.log(air['#Passengers'])
air_log

# %%
# plot the log transformed data with the rolling mean and std

air_log['rolling_mean12'] = air_log['#Passengers'].rolling(12).mean()
air_log['rolling_std12'] = air_log['#Passengers'].rolling(12).std()
air_log.plot()

# %% [markdown]
# - after taking the log, the variance is become constant
# - the rolling mean is trending upwards, but I will take different levels of diff to address this

# %%
# use different levels of difference and check if it is stationary

# trying 1 level of diff
# make a copy of the dataframe
air_log_diff = air_log.copy()

air_log_diff['#Passengers'] = air_log_diff['#Passengers'].diff()
air_log_diff['rolling_mean12'] = air_log_diff['#Passengers'].rolling(12).mean()
air_log_diff['rolling_std12'] = air_log_diff['#Passengers'].rolling(12).std()

air_log_diff.plot()

# %% [markdown]
# - After taking the log and 1 level of diff, both the rolling mean and std are constant over time.
# - However, there still appears to be seasonality in the data

# %%
# trying 2 levels of diff
# make a copy of the dataframe
air_log_diff_diff = air_log.copy()

air_log_diff_diff['#Passengers'] = air_log_diff_diff['#Passengers'].diff().diff()
air_log_diff_diff['rolling_mean12'] = air_log_diff_diff['#Passengers'].rolling(12).mean()
air_log_diff_diff['rolling_std12'] = air_log_diff_diff['#Passengers'].rolling(12).std()

air_log_diff_diff.plot()

# %% [markdown]
# - After the log and 2 levels of difference, the rolling mean is again, constant. The standard deviation is also constant, with fluctuations throughout.
# - There is still seasonality that needs to be addressed.

# %%
# trying 3 levels of diff
# make a copy of the dataframe
air_log_diff_diff_diff = air_log.copy()

air_log_diff_diff_diff['#Passengers'] = air_log_diff_diff_diff['#Passengers'].diff().diff().diff()
air_log_diff_diff_diff['rolling_mean12'] = air_log_diff_diff_diff['#Passengers'].rolling(12).mean()
air_log_diff_diff_diff['rolling_std12'] = air_log_diff_diff_diff['#Passengers'].rolling(12).std()

air_log_diff_diff_diff.plot()

# %% [markdown]
# - With log, 3 levels of diff does not offer much improvement from 2 levels of diff.

# %%
# trying sqrt instead of log
# make a copy
air_sqrt = air.copy()

air_sqrt['#Passengers'] = np.sqrt(air['#Passengers'])
air_sqrt['rolling_mean12'] = air_sqrt['#Passengers'].rolling(12).mean()
air_sqrt['rolling_std12'] = air_sqrt['#Passengers'].rolling(12).std()
air_sqrt.head(15)

# %%
# plot the sqrt transformed data

air_sqrt.plot()

# %% [markdown]
# - the variance has become more constant, however, there is still a slight upward trend.
# - the heteroscedasticity was not removed as well as through the log transformation

# %%
# trying cube root instead of log
# make a copy
air_cbrt = air.copy()

air_cbrt['#Passengers'] = np.cbrt(air_cbrt['#Passengers'])
air_cbrt['rolling_mean12'] = air_cbrt['#Passengers'].rolling(12).mean()
air_cbrt['rolling_std12'] = air_cbrt['#Passengers'].rolling(12).std()
air_cbrt.head(15)

# %%
# plot the data after cube root transformation

air_cbrt.plot()

# %% [markdown]
# - after taking the cube root, the variance became even more constant than when taking the square root
# - however, there is still a slight hint of an upward trend, so it did not do as well as the log at making the variance more consistent.

# %%
# the best combination of transforming and taking the difference to make the data stationary is:
# 1. take the log
# 2. take 2 levels of diff

air_log_diff2 = air.copy()

air_log_diff2['#Passengers'] = np.log(air['#Passengers']).diff().diff()
air_log_diff2['rolling_mean12'] = air_log_diff2['#Passengers'].rolling(12).mean()
air_log_diff2['rolling_std12'] = air_log_diff2['#Passengers'].rolling(12).std()

air_log_diff2.plot()

# %% [markdown]
# ### 4. Get the p-value from Augmented Dickey-Fuller test to make the data stationary.

# %%
# use adfuller to check if the data is stationary
# drop NaNs

results = adfuller(air_log_diff2['#Passengers'].dropna())
results

# %% [markdown]
# - the p-value is much smaller than 0.05, so we can confidently reject the null hypothesis that the data is non-stationary. The data is stationary.

# %% [markdown]
# ### Visually check for seasonality

# %%
# plot the data that was log transformed and 2 levels of differences taken

air_log_diff2.plot()

# %% [markdown]
# - the ups and downs seems to be cyclical and repeat every 12 months (1 year)

# %%
# try taking a third level of diff to remove seasonality
# knowing that the seasonality cycles over a period of 12 months, will take the diff with period = 12

air_log_diff2_diff12 = air_log_diff2.diff(12)
air_log_diff2_diff12.plot()

# %% [markdown]
# - After taking a third level of diff with period = 12, the ups and downs are more random and do not have as much pattern to them.

# %% [markdown]
# ### Using seasonal decompose to check the auto covariance

# %%
# use seasonal decompose to check the original data

results = seasonal_decompose(air['#Passengers'])
results.plot()

# %% [markdown]
# - there is trend and seasonality in the original data

# %%
# use seasonal decompose to check the auto covariance of the transformed data

results = seasonal_decompose(air_log_diff2_diff12['#Passengers'].dropna())
results.plot()

# %% [markdown]
# - in the transformed data, the trend has been removed, but there is still seasonality

# %%
# get the seasonality from seasonal

seasonal = results.seasonal

# remove seasonality using seasonal

air_log_diff2_diff12_deco = air_log_diff2_diff12.copy()
air_log_diff2_diff12_deco['#Passengers'] = (air_log_diff2_diff12['#Passengers'] - seasonal)
air_log_diff2_diff12_deco['#Passengers'].plot()

# %%
# check the seasonality after doing seasonal decompose

results2 = seasonal_decompose(air_log_diff2_diff12_deco['#Passengers'].dropna())
results2.plot()

# %% [markdown]
# - seasonal decompose still indicates there is seasonality, but looking at the plot of the data, there is no longer an obvious pattern.
