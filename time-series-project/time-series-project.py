# %% [markdown]
# # Time Series Project - Predicting Stock Price

# %%
# install yahoo finance

# %pip install yfinance

# %%
# import

import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from scipy.signal import find_peaks, periodogram

from tabulate import tabulate

import sqlite3

# %% [markdown]
# # Preparing the Data

# %%
# set the ticker and date frame

ticker = "DAL"
start_date = "2022-04-01"
end_date = "2025-03-31"

# %%
# get the data

data = yf.download(ticker, start=start_date, end=end_date)

# %%
# check the data

data

# %%
# remove multi-index

data = data.droplevel(level=1, axis=1).reset_index()
data

# %%
# create a new dataframe with just the date and close price

close = pd.DataFrame(columns=['Date', 'Close'])
close['Date'] = data['Date']
close['Close'] = data['Close']
close

# %%
# make sure the data type of the Date column is datetime

close.info()

# %%
# make the datetime the index

close = close.set_index('Date')
close

# %%
# auto detect the frequency of the datetime

inferred_freq = pd.infer_freq(close.index)
print(inferred_freq)

# %%
# manually set the frequency to business days, and fill missing dates with forward fill

close = close.asfreq('B')
close['Close'] = close['Close'].fillna(method='ffill')
close.info()

# %% [markdown]
# # Checking if the Data is Stationary

# %%
# plot the close prices

close.plot()

# %% [markdown]
# - the plot of the close prices show an upward trend in prices over time
# - there is a vague pattern to the price increases and decreases, although it is difficult to tell if they repeat in a consistent manner or over what length of period of time
# - I will calculate the rolling mean and standard deviations to confirm if the data is stationary

# %%
# calculate the rolling mean and rolling std

close['rolling_mean'] = close['Close'].rolling(120).mean()
close['rolling_std'] = close['Close'].rolling(120).std()

close.plot()

# %% [markdown]
# - the rolling mean shows a positive trend in the close price
# - the rolling standard deviation also signals that the standard deviation is increasing over time
# - based on the above, the data is not stationary. I will confirm this by using the ADFuller test
# - the null hypothesis of the ADFuller test is that the data is not stationary, I would need a p value of less 0.05 in order to reject my null hypothesis

# %%
# use the ADFuller test to confirm that the data is not stationary

adfuller(close['Close'])

# %% [markdown]
# - the p-value of the ADF test is much greater than 0.05, which means I cannot reject the null hypothesis that the data is not stationary.
# - to make the data more stationary, I will remove non-constant variance by taking the log the close price, and then remove the trend by taking the difference of the logged close price.

# %% [markdown]
# # Making the Data Stationary

# %%
# create a copy of the dataframe with the log of the close price

close_log = close.copy()
close_log['Close'] = np.log(close_log['Close'])
close_log['rolling_mean'] = close_log['Close'].rolling(30).mean()
close_log['rolling_std'] = close_log['Close'].rolling(30).std()

close_log.plot()

# %% [markdown]
# - after taking the log of the close price, the variance is more consistent, but still increases over time
# - the rolling mean is also more stable, but there is still a slight upward trend

# %%
# check the adfuller p-value for the logged close price

adfuller(close_log['Close'])

# %% [markdown]
# - the p-value of the logged close price is again greater than 0.05, this means that I still cannot reject the null hypothesis that the data is not stationary. I will continue to make the data more stationary by taking different levels of differencing of the logged data.

# %%
# create a copy of the dataframe with the log of the close price and 1 level of diff

close_log_diff = close.copy()
close_log_diff['Close'] = np.log(close_log_diff['Close']).diff()
close_log_diff['rolling_mean'] = close_log_diff['Close'].rolling(30).mean()
close_log_diff['rolling_std'] = close_log_diff['Close'].rolling(30).std()

close_log_diff.plot()

# %% [markdown]
# - the rolling mean and the rolling standard deviation both no longer have an obvious upward trend.
# - I will check the adfuller p-value to see if the null hypothesis can be rejected or not.

# %%
# check the adfuller p-value on the logged close price with 1 level of difference

adfuller(close_log_diff['Close'].dropna())

# %% [markdown]
# - the p-value is very small, this means that we can reject the null hypothesis that the data is non-stationary.
# - I will continue to check additional levels of differencing

# %%
# create a copy of the dataframe with the log of the close price and 2 levels of diff

close_log_diff_diff = close.copy()
close_log_diff_diff['Close'] = np.log(close_log_diff_diff['Close']).diff().diff()
close_log_diff_diff['rolling_mean'] = close_log_diff_diff['Close'].rolling(30).mean()
close_log_diff_diff['rolling_std'] = close_log_diff_diff['Close'].rolling(30).std()

close_log_diff_diff.plot()

# %% [markdown]
# - at 2 levels of differencing, the rolling mean shows no trend at all, and the standard deviation does not show any overarching trends of increase or decrease.
# - I will check the adfuller p-value to confirm.

# %%
# check the adfuller p-value

adfuller(close_log_diff_diff['Close'].dropna())

# %% [markdown]
# - the p-value is very small, but greater than at 1 level of diff. Again, we can reject the null hypothesis that the data is not stationary.
# - since the p-value increased with 2 levels of diff, it means that the data already became stationary at 1 level of diff, and that additional levels of diff may or may not be unnecessary.

# %% [markdown]
# # Check for Seasonality
#
# Differencing helps to alleviate seasonality, but the ADFuller test does not check for seasonality, so I will double check that there is no more seasonality left in the transformed data by using the following methods:
# 1. Seasonal decompose to visualize seasonality
# 2. Calculate the period of seasonality using periodogram

# %%
# use seasonal decompose to visualize any seasonality in the data

seasonal_results = seasonal_decompose(close_log_diff['Close'].dropna())
seasonal_results.plot()

# %% [markdown]
# - The seasonal plot shows a repeating pattern, which indicates that there may be seasonality in the data.
# - I will next try to calculate the length of the seasonal period using perdiogram

# %%
# calculate the periods using periodogram

# compute the power spectrum
frequencies, power = periodogram(close_log['Close'].dropna())

# find peaks in the power specturm
peaks, _ = find_peaks(power, height=0.1 * max(power))

# convert frequency to period
periods = (1 / frequencies[peaks])

# return the period
periods

# %% [markdown]
# - the periods that the periodogram found significant are 781 and 195.25 business days.
# - 781 is the length of the data, so this indicates that there is an overarching trend across the entire time period
# - 195.25 business days translate to roughly 275 days on the standard calendar, which is 3 quarters of a year
# - I will next plot the periods to visualize the peaks in the frequencies

# %%
# plot the periods (1/frequencies)

plt.plot((1/frequencies), power)
plt.title("Periodogram")
plt.xlabel("Periods (# Business Days)")
plt.ylabel("Power")
plt.axvline(x=195.25, color='red');
plt.axvline(x=260, color='purple');

# %% [markdown]
# - the above plot shows the frequencies that were found by the power spectrum.
# - there is a peak around 200, which correlates to the 195.25 found previously
# - after 260, the power continues to increase, this is likely due to a trend in the data (which will be addressed through differencing)
# - I will remove the seasonality next

# %% [markdown]
# # Removing Seasonality

# %%
# save the seasonal element from the results

seasonal = seasonal_results.seasonal

# remove the seasonality by subtracting it from the close prices

close_log_season = close_log.copy()
close_log_season['Close'] = (close_log['Close'] - seasonal)
close_log_season['Close'].plot()

# %% [markdown]
# ### Analyze the ACF and PACF plots for best p, d, q

# %%
# plot the acf with the log-transformed data and seasonality removed

plot_acf(close_log_season['Close'].dropna())

# %% [markdown]
# - the gradually decreasing ACF plot indicates autoregressive characteristics which means that there are predictable patterns based on past values. This confirms that I should consider AR terms in my model.

# %%
# plot the acf with 1 level of diff

plot_acf(close_log_season['Close'].diff().dropna())

# %% [markdown]
# - The ACF shows no significant autocorrelation after lag of 0.
# - This suggests that when using 1 level of diff the MA component in the ARIMA model may be 0 (q = 0).

# %%
# plot the acf with 2 levels of diff

plot_acf(close_log_season['Close'].diff().diff().dropna())

# %% [markdown]
# - The ACF shows no significant autocorrelation after lag of 1.
# - This suggests that when using 2 levels of diff, the MA component in the ARIMA model may be 1 (q = 1).

# %%
# plot the acf with 3 levels of diff

plot_acf(close_log_season['Close'].diff().diff().diff().dropna())

# %% [markdown]
# - The ACF shows no significant autocorrelation after lag of 1.
# - This suggests that when using 3 levels of diff, the MA component in the ARIMA model may be 2 (q = 2).

# %%
# plot the pacf

plot_pacf(close_log_season['Close'].dropna())

# %% [markdown]
# - With no differencing, the PACF shows a significant drop after lag 1, suggesting that the AR component in the ARIMA model may be 1 (p = 1).

# %%
# plot the pacf with 1 level of diff

plot_pacf(close_log_season['Close'].diff().dropna())

# %% [markdown]
# - With 1 level of differencing, the PACF shows no significant autocorrelation after lag 0, suggesting that the AR component in the ARIMA model may be 0 (p = 0).

# %%
# plot the pacf with 2 levels of diff

plot_pacf(close_log_season['Close'].diff().diff().dropna())

# %% [markdown]
# - With 2 levels of differencing, the PACF shows significant autocorrelation up to lag 8, suggesting that the AR component in the ARIMA model may be 8 (p = 8).

# %%
# plot the pacf with 3 levels of diff

plot_pacf(close_log_season['Close'].diff().diff().diff().dropna())

# %% [markdown]
# - With 3 levels of differencing, the PACF shows significant autocorrelation up to lag 17, suggesting that the AR component in the ARIMA model may be 17 (p = 17).

# %%
# confirming that the data is not non-stationary using adfuller

final_transform_adf = adfuller(close_log_season['Close'].diff().diff().dropna())
final_transform_adf

# %% [markdown]
# ### Based on the ACF and PACF plots, my best hyperparameters are q=1, d=2, p=8

# %% [markdown]
# # ARIMA Model

# %%
# train test split

train, test = train_test_split(close_log_season['Close'].dropna(), test_size=60, shuffle=False)

display(train.head())
display(train.tail())
display(test.head())
display(test.tail())

# %%
# set my best hyperparameters based on the acf and pacf plots

q = 1
d = 2
p = 8

# define the ARIMA model

model8_2_1 = ARIMA(train, order=(p,d,q))
results_model8_2_1 = model8_2_1.fit()

# check the results summary

results_model8_2_1.summary()

# %% [markdown]
# The SARIMAX results:
# 1. Ljung-Box prob(Q) is high, which means that there are no significant autocorrelations left in the residuals
# 2. Jarque-Bera prob(JB) is low, indicating that residuals are not normally distributed, which can affect confidence intervals, but forecasting models may still work well
# 3. Heteroskedasticity prob(H) is also low, which indicates that heteroskedasticity is still present
#
# - Next, I will evaluate the model by predicting test and calculating the RMSE

# %%
# get the predictions

train_preds_model8_2_1 = results_model8_2_1.fittedvalues

# build a df to store predictions

results_model8_2_1_train = pd.DataFrame(columns=['train','train_preds'])
results_model8_2_1_train['train'] = np.exp(train)
results_model8_2_1_train['train_preds'] = np.exp(train_preds_model8_2_1)
results_model8_2_1_train.plot(alpha = 0.5)
plt.xlim(train.index[-60], train.index[-1])

# %%
# forecasting test

forecast_60_days = results_model8_2_1.forecast(steps=len(test))
forecast_60_days

test_results_model8_2_1 = pd.DataFrame(columns=['test','test_pred'])
test_results_model8_2_1['test'] = np.exp(test)
test_results_model8_2_1['test_pred'] = np.exp(forecast_60_days)
test_results_model8_2_1.plot()

# %%
# calculate the metrics for the model

model8_2_1_train_rmse = root_mean_squared_error(results_model8_2_1_train['train'], results_model8_2_1_train['train_preds'])
model8_2_1_test_rmse = root_mean_squared_error(test_results_model8_2_1['test'], test_results_model8_2_1['test_pred'])

print('Train RMSE', model8_2_1_train_rmse)
print('Test RMSE', model8_2_1_test_rmse)

# %% [markdown]
# - The RMSE is lower on train than on test, this indicates the model is overfit
# - I will try to improve the model by using AutoARIMA to search for the best hyperparameters

# %%
# create a dataframe to store all of the best models from each method

best_models = pd.DataFrame(columns=['model', 'hyperparameters', 'train RMSE', 'test RMSE'])

new_row_arima8_2_1 = pd.DataFrame([{'model': 'ARIMA using acf and pacf',
                                    'hyperparameters': 'p=8, d=2, q=1',
                                    'train RMSE': model8_2_1_train_rmse,
                                    'test RMSE': model8_2_1_test_rmse}])
best_models = pd.concat([best_models, new_row_arima8_2_1], ignore_index=True)

best_models

# %% [markdown]
# # Using AutoARIMA to search for the best hyperparameters

# %%
# import pmdarima

import pmdarima as pm

# %%
# set the search parameters

pmarima = pm.AutoARIMA(start_p=0, max_p=10, start_d=0, max_d=3, start_q=0, max_q=5, seasonal=False, random_state=42,
                       stepwise=True, suppress_warnings=True, error_action='ignore', trace=True)

# %%
# fit the model

pmarima.fit(train)

# get the best model
model=pmarima.model_
model.summary()

# %% [markdown]
# - the best model from the autoARIMA search is (0,1,0), which only uses differencing, and no autoregressive or moving average terms.

# %%
# get the predictions from the best model

pm_test_pred, confs = model.predict(n_periods=len(test), return_conf_int=True)

# plot the predictions

plt.plot(np.exp(test), label='test')
plt.plot(np.exp(pm_test_pred), label='pred')
plt.fill_between(test.index, np.exp(confs[:,0]), np.exp(confs[:,1]), color='blue', alpha=.1)
plt.legend()

# %% [markdown]
# - the model is predicting a constant price

# %%
# store the test predictions

pm_test_preds = pd.DataFrame(columns=['test','test_preds'])
pm_test_preds['test'] = np.exp(test)
pm_test_preds['test_preds'] = np.exp(pm_test_pred)

# %%
# get the train predictions

pm_train_pred = model.predict_in_sample()

# %%
# store the train predictions

pm_train_preds = pd.DataFrame(columns=['train','train_preds'])
pm_train_preds['train'] = np.exp(train)
pm_train_preds['train_preds'] = np.exp(pm_train_pred)

# %%
# calculate the metrics for the model

arima010_train_rmse = root_mean_squared_error(pm_train_preds['train'], pm_train_preds['train_preds'])
arima010_test_rmse = root_mean_squared_error(pm_test_preds['test'], pm_test_preds['test_preds'])

print('Train RMSE', arima010_train_rmse)
print('Test RMSE', arima010_test_rmse)

# %% [markdown]
# - the RMSE for the autoARIMA's best model is lower than the RMSE of the model with manually select hyperparameters. However, the autoARIMA's best hyperparameters is basically just predicting the previous day, it is a very simple model and does not include auto-regression or moving average.
# - instead of using autoARIMA to find the best model, I will do a manual loop to find the best hyperparameters that give me the best test RMSE

# %%
# add results to best model dataframe

new_row_arima0_1_0 = pd.DataFrame([{'model': 'ARIMA using autoARIMA',
                                    'hyperparameters': 'p=0, d=1, q=0',
                                    'train RMSE': arima010_train_rmse,
                                    'test RMSE': arima010_test_rmse}])
best_models = pd.concat([best_models, new_row_arima0_1_0], ignore_index=True)

best_models

# %% [markdown]
# # Manual loop over ARIMA

# %%
# create a function that accepts different values of p, d and q and outputs the train RMSE and test RMSE

def arima_search(train, test, p, q, d):
  arima_model = ARIMA(train, order=(p,d,q))
  arima_result = arima_model.fit()

  # get the train predictions
  train_pred = arima_result.fittedvalues

  # build a df to store train predictions
  train_preds = pd.DataFrame(columns=['train','train_pred'])
  train_preds['train'] = np.exp(train)
  train_preds['train_pred'] = np.exp(train_pred)

  # get the test predictions
  test_pred = arima_result.forecast(steps=len(test))

  # build a df to store test predictions
  test_preds = pd.DataFrame(columns=['test','test_pred'])
  test_preds['test'] = np.exp(test)
  test_preds['test_pred'] = np.exp(test_pred)

  train_RMSE = root_mean_squared_error(train_preds['train'], train_preds['train_pred'])
  test_RMSE = root_mean_squared_error(test_preds['test'], test_preds['test_pred'])

  return train_RMSE, test_RMSE

# %%
# loop over a ranges of p, q, and d and store the train and test RMSE to a list

from itertools import product

p_range = list(range(0,6))
q_range = list(range(0,6))
d_range = list(range(1,3))

ARIMA_results = []

for p, q, d in product(p_range, q_range, d_range):
  try:
    train_rmse, test_rmse = arima_search(
      train, test,
      p=p,
      q=q,
      d=d
    )
    ARIMA_results.append({
      'p': p,
      'd': d,
      'q': q,
      'train_rmse': train_rmse,
      'test_rmse': test_rmse
    })
  except Exception as e:
    print(f"Failed for p={p}, d={d}, q={q} — Error: {e}")


# %%
# turn the results list into a dataframe, then sort, with the best test RMSE on top
ARIMA_results_df = pd.DataFrame(ARIMA_results)

best_arima_loop = ARIMA_results_df.sort_values(by='test_rmse').reset_index(drop=True)
best_arima_loop.head(10)

# %% [markdown]
# - The best test RMSE is at p=5, d=2, q=1

# %%
# transform the dataframe to markdown

print(tabulate(best_arima_loop.head(10), tablefmt="pipe", headers="keys", showindex=False))

# %%
# plot the best model from the manual ARIMA search

arima_loop = ARIMA(train, order=(5,2,1))
arima_loop_results = arima_loop.fit()

# get the train predictions
arima_loop_train_pred = arima_loop_results.fittedvalues

# build a df to store train predictions
arima_loop_train_preds = pd.DataFrame(columns=['train','train_pred'])
arima_loop_train_preds['train'] = np.exp(train)
arima_loop_train_preds['train_pred'] = np.exp(arima_loop_train_pred)

# get the test predictions
arima_loop_test_pred = arima_loop_results.forecast(steps=len(test))

# build a df to store test predictions
arima_loop_test_preds = pd.DataFrame(columns=['test','test_pred'])
arima_loop_test_preds['test'] = np.exp(test)
arima_loop_test_preds['test_pred'] = np.exp(arima_loop_test_pred)

arima_loop_train_RMSE = root_mean_squared_error(arima_loop_train_preds['train'], arima_loop_train_preds['train_pred'])
arima_loop_test_RMSE = root_mean_squared_error(arima_loop_test_preds['test'], arima_loop_test_preds['test_pred'])

arima_loop_test_preds.plot()


# %%
# add the results to the best model dataframe for comparison later

new_row_arima_loop = pd.DataFrame([{'model': 'ARIMA search loop',
                                    'hyperparameters': 'p=5, d=2, q=1',
                                    'train RMSE': arima_loop_train_RMSE,
                                    'test RMSE': arima_loop_test_RMSE}])
best_models = pd.concat([best_models, new_row_arima_loop], ignore_index=True)

best_models

# %% [markdown]
# # Modeling using Prophet

# %%
from prophet import Prophet

# create a model
m = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=15.0)
m.add_seasonality(name='custom_seasonality', period=195, fourier_order=5)
prophet_train = np.exp(train).reset_index()

# rename the columns to ds and y
prophet_train.columns = ['ds', 'y']
prophet_train

# fit the data
m.fit(prophet_train)

# %% [markdown]
# - In a prophet model, changepoint_prior_scale and seasonality_prior_scale control the flexibility of how much the model is allowed to bend to fit the data.
# - A low value in changepoint_prior_scale means that the trend line is more stable, which is good for when the data that has noise and we just want to capture the overall direction. A high value means that the model will allow sudden jumps in trend, so that is can account for real disruptions.
# - A low value in seasonality_prior_scale means that the seasonality is smooth and simple, while a high value allows for more complex seasonal patterns.
# - I am choosing a high seasonality_prior_scale, because the seasonality in the data is complicated and does not have a clear, simple, pattern.
# - I am choosing a high changepoint_prior_scale as well because the trend is not consistent, it involves frequent fluctuation.
# - add_seasonality allows me to define a custom seasonal component beyond the default ones, which are yearly, weekly and daily. I am defining the custom seasonal component to the peak that was discovered by the periodogram, which was 195 days.
# - The fourier_order controls how complex the seasonal component is allowed to be. It utilizes sine and cosine terms to model the seasonal cycle. A higher value means that seasonal patterns can be flexible and detailed, while a lower values means smoother and simpler seasonality. Low values are between 3-5, medium values range from 10-15, and high values are over 20.
# - I am selecting a low fourier_order, because stock prices are subject to many external factors, and I do not want the model to overfit on the noise.

# %%
# make a future dataframe

future = m.make_future_dataframe(periods=len(test), freq='B')
forecast = m.predict(future)
m.plot(forecast)
# plt.xlim(test.index[0], test.index[-1])
plt.plot(np.exp(test), label='test', color='red');

# %% [markdown]
# - unlike the ARIMA model, the predictions are not a straight line, and they seem to continue the up and down pattern that is historically seen in this particular stock.

# %%
# predict the train
train_forecast = m.predict(prophet_train[['ds']])

# merge actual (exp(train)) and predicted (yhat) by 'ds'
m_train_preds = prophet_train[['ds']].copy()
m_train_preds['train'] = np.exp(train.values)
m_train_preds = m_train_preds.merge(train_forecast[['ds', 'yhat']], on='ds', how='left')
m_train_preds.rename(columns={'yhat': 'm_train_pred'}, inplace=True)

# %%
# save the predictions for only the test time period
m_test_preds = [forecast.loc[forecast['ds'] == month, 'yhat'].values[0] for month in test.index]

# save the predictions in a dataframe
m_test_results = pd.DataFrame(columns = ['test', 'm_test_pred'])
m_test_results['test'] = np.exp(test)
m_test_results['m_test_pred'] = m_test_preds

# get the metrics for the Prophet model
print('Train RMSE', root_mean_squared_error(m_train_preds['train'], m_train_preds['m_train_pred']))
print('Test RMSE', root_mean_squared_error(m_test_results['test'], m_test_results['m_test_pred']))

# %% [markdown]
# - The RMSE for my Prophet model with manually selected hyperparameters is higher than that of the autoARIMA model
# - The train RMSE is much lower than that of test, which indicates that it is extremely overfit
# - I will use a loop to find the best hyperparameters for the Prophet model

# %%
# create a function to try different hyperparameters in the prophet model and then loop over it

def prophet_search(changepoint_prior_scale=0.5, seasonality_prior_scale=15.0, period=195, fourier_order=5):
  # create a model
  m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
  m.add_seasonality(name='custom_seasonality', period=period, fourier_order=fourier_order)
  prophet_train = np.exp(train).reset_index()

  # rename the columns to ds and y
  prophet_train.columns = ['ds', 'y']
  prophet_train

  # fit the data
  m.fit(prophet_train)

  future = m.make_future_dataframe(periods=len(test), freq='B')
  forecast = m.predict(future)

  # predict the train
  train_forecast = m.predict(prophet_train[['ds']])

  # merge actual (exp(train)) and predicted (yhat) by 'ds'
  m_train_preds = prophet_train[['ds']].copy()
  m_train_preds['train'] = np.exp(train.values)
  m_train_preds = m_train_preds.merge(train_forecast[['ds', 'yhat']], on='ds', how='left')
  m_train_preds.rename(columns={'yhat': 'm_train_pred'}, inplace=True)

  # save the predictions for only the test time period
  m_test_preds = [forecast.loc[forecast['ds'] == month, 'yhat'].values[0] for month in test.index]

  # save the predictions in a dataframe
  m_test_results = pd.DataFrame(columns = ['test', 'm_test_pred'])
  m_test_results['test'] = np.exp(test)
  m_test_results['m_test_pred'] = m_test_preds

  train_RMSE = root_mean_squared_error(m_train_preds['train'], m_train_preds['m_train_pred'])
  test_RMSE = root_mean_squared_error(m_test_results['test'], m_test_results['m_test_pred'])

  return train_RMSE, test_RMSE

# %%
# test the function

train_RMSE, test_RMSE = prophet_search(changepoint_prior_scale=0.1, seasonality_prior_scale=20.0, period=195, fourier_order=4)
train_RMSE, test_RMSE

# %%
# define the hyperparameters ranges

changepoint_prior_scales = [0.05, 0.1, 0.2, 0.3]
seasonality_prior_scales = [5.0, 10.0, 20.0]
fourier_orders = [3, 4, 5]
periods = [195]

# %%
# use itertools.product to do a search over the hyperparameters

from itertools import product

prophet_results = []

for cps, sps, fo, p in product(changepoint_prior_scales, seasonality_prior_scales, fourier_orders, periods):
  try:
    train_rmse, test_rmse = prophet_search(
      changepoint_prior_scale=cps,
      seasonality_prior_scale=sps,
      period=p,
      fourier_order=fo
    )
    prophet_results.append({
      'changepoint_prior_scales': cps,
      'seasonality_prior_scales': sps,
      'fourier_orders': fo,
      'period': p,
      'train_rmse': train_rmse,
      'test_rmse': test_rmse
    })
  except Exception as e:
    print(f"Failed for cps={cps}, sps={sps}, fo={fo}, p={p} — Error: {e}")


# %%
prophet_results_df = pd.DataFrame(prophet_results)

# Sort by test RMSE
best_prophet = prophet_results_df.sort_values(by='test_rmse').reset_index(drop=True)
best_prophet.head(10)

# %% [markdown]
# Based on the Prophet model search, the best hyperparameters are:
# 1. changepoint_prior_scales = 0.5
# 2. seasonality_prior_scales = 10.0
# 3. fourier_orders = 4
# 4. when period is set to 195

# %% [markdown]
# - Next, I will use the best hyperparameters in the Prophet model to predict future prices

# %%
# add the results to the best model dataframe for comparison later

new_row_prophet = pd.DataFrame([{'model': 'Prophet search loop',
                                    'hyperparameters': 'cps=0.5, sps=10.0, f=4, p=195',
                                    'train RMSE': best_prophet.loc[0, 'train_rmse'],
                                    'test RMSE': best_prophet.loc[0, 'test_rmse']}])
best_models = pd.concat([best_models, new_row_prophet], ignore_index=True)

best_models

# %%
# convert the dataframe to markdown

print(tabulate(best_models, tablefmt="pipe", headers="keys", showindex=False))

# %% [markdown]
# # Predicting prices for the future

# %% [markdown]
# ### 1. Predicting using the best Prophet model

# %%
# create a new model with the best hyperparameters
m_best = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10.0)
m_best.add_seasonality(name='custom_seasonality', period=195, fourier_order=4)
prophet_all = np.exp(close_log_season['Close']).dropna().reset_index()

# rename the columns to ds and y
prophet_all.columns = ['ds', 'y']
prophet_all

# fit the data
m_best.fit(prophet_all)

# forecast the future
future = m_best.make_future_dataframe(periods=10, freq='B')
forecast = m_best.predict(future)

# check forecast tail
forecast.tail()

# %%
# plot all the predictions

m_best.plot(forecast)
plt.plot(np.exp(test), label='test', color='red');

# %%
# plot the last 30 predictions, which includes future predictions as well

last_30 = forecast.iloc[-30:]

m_best.plot(forecast)
plt.xlim(last_30['ds'].min(), last_30['ds'].max())
plt.plot(np.exp(test), label='test', color='red');

# %% [markdown]
# # Save the predictions to a database

# %%
# save the future predictions to a dataframe

prophet_preds = forecast[['ds','yhat']].copy()
prophet_preds_future = prophet_preds[-9:]
prophet_preds_future

# %%
# convert the dataframe to markdown

print(tabulate(prophet_preds_future, tablefmt="pipe", headers="keys", showindex=False))

# %%
# save the predictions to csv

prophet_preds_future.to_csv('prophet_predictions.csv', index=True)

# %%
# create a new sqlite connection

conn = sqlite3.connect('yfinance.db')

# %%
# save the predictions to sql database

prophet_preds_future.to_sql('prophet_predictions', conn, index=True, if_exists='replace')

# %%
# read from sql table to double check

future_preds_from_sql = pd.read_sql('SELECT * FROM prophet_predictions', conn)
future_preds_from_sql

# %%
# close the sql connection

conn.close()
