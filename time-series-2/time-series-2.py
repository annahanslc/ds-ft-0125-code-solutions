# %% [markdown]
# # time-series-2

# %%
# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# import data

df = pd.read_csv('AirPassengers.csv')
df.head()

# %% [markdown]
# ### Preparing the Data

# %%
# prepare data: change to datetime

df['Month'] = pd.to_datetime(df['Month'])
df.info()

# %%
# change index to the datetime

df.set_index('Month', inplace=True)
df.head()

# %%
# set the frequency to Month

df = df.resample('ME').sum()
df.index.freq = 'ME'
df.info()

# %%
# plot the data to visualize

df.plot()

# %% [markdown]
# ### 1. Use “plot_pacf” and “plot_acf” to get the “p” and “q” values respectively.

# %%
# import plotting tools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
# plot the acf with original data

plot_acf((df['#Passengers']))

# %% [markdown]
# - there is seasonality, and the increases and decreases are geometric
# - not a good indicator of q, try different transformations

# %%
# plot the acf with only log

plot_acf(np.log(df['#Passengers']).dropna())

# %% [markdown]
# - again, the increases and decreases are geometric with no sudden drop
# - still does not indicate a good q value

# %%
# plot the acf

plot_acf(np.log(df['#Passengers']).diff().dropna())

# %% [markdown]
# - with log transform and 1 levels of diff, the drop appears after 2
# - so best q=2

# %%
# plot the acf

plot_acf(np.log(df['#Passengers']).diff().diff().dropna())

# %% [markdown]
# - with log transform and 2 levels of diff, the drop appears after 3
# - so best q=3

# %%
# plot the acf

plot_acf(np.log(df['#Passengers']).diff().diff().diff().dropna())

# %% [markdown]
# - with log transform and 3 levels of diff, the drop appears after 2
# - so best q=2

# %%
# use pacf to check for best p
# on original data

plot_pacf(df['#Passengers'].dropna())

# %% [markdown]
# - on the un-transformed data, the biggest drop occurs after 2, but 3 is still significant
# - so the best p = 3

# %%
# use pacf to check for best p
# on log-transformed data

plot_pacf(np.log(df['#Passengers']).dropna())

# %% [markdown]
# - on log transformed data, the best p = 2

# %%
# use pacf to check for best p
# on log-transformed data with 1 level of diff

plot_pacf(np.log(df['#Passengers']).diff().dropna())

# %% [markdown]
# - on log transformed data with 1 level of diff, best p either at 2 or 3

# %%
# use pacf to check for best p
# on log-transformed with 2 levels of diff

plot_pacf(np.log(df['#Passengers']).diff().diff().dropna())

# %% [markdown]
# - on log transformed data with 2 levels of diff, the best p = 3

# %%
# use pacf to check for best p
# on log-transformed with 3 levels of diff data

plot_pacf(np.log(df['#Passengers']).diff().diff().diff().dropna())

# %% [markdown]
# - on log transformed data with 3 levels of diff, best p = 3

# %% [markdown]
# ### Best parameters q = 3, p = 3, d = 2

# %% [markdown]
# ### 2. Build an ARIMA model based on the “p” and “q” values obtained from above and get the RMSE.

# %%
# take the log of the data

df_log = df.copy()
df_log['#Passengers'] = np.log(df['#Passengers'])
df_log.head()

# %%
# import train test split

from sklearn.model_selection import train_test_split

# %%
# define train test split

train, test = train_test_split(df_log, test_size=12, shuffle=False)

# %%
# check that the end of the train is where the test starts

display(train.tail())
display(test.head())

# %%
# import ARIMA

from statsmodels.tsa.arima.model import ARIMA

# %%
# define the ARIMA model and parameters

q=3
d=2
p=3

model_3_2_3 = ARIMA(train, order=(p, d, q))
results = model_3_2_3.fit()
results.summary()

# %%
# check the fitted values with the actual values

train_preds = results.fittedvalues

results_df = pd.DataFrame(columns = ['train', 'train_preds'])
results_df['train'] = np.exp(train)
results_df['train_preds'] = np.exp(train_preds)
results_df.plot(alpha = 0.5)
plt.xlim(train.index[-30], train.index[-1])

# %% [markdown]
# - the predictions are basically copying the results from the previous month

# %%
# forecasting the test

forecast_12_months = results.forecast(steps=len(test))
forecast_12_months

# %%
# store the test results

test_results = pd.DataFrame(columns=['test','test_pred'])
test_results['test'] = np.exp(test)
test_results['test_pred'] = np.exp(forecast_12_months)

test_results

# %%
# plot the test predictions against the actual test values

test_results.plot()

# %%
# evaluate the model using metrics
# imports

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

# %%
# check the metrics

print('MSE', mean_squared_error(test_results['test'], test_results['test_pred']))
print('RMSE', root_mean_squared_error(test_results['test'], test_results['test_pred']))
print('MAPE', mean_absolute_percentage_error(test_results['test'], test_results['test_pred']))

# %% [markdown]
# ### The RMSE is 79.93937704939401

# %% [markdown]
# ### Using PMDARIMA to find the best parameters and account for seasonality

# %%
# import

import pmdarima as pm

# %%
# set the search parameters

pmarima = pm.AutoARIMA(start_p=0, max_p=5, start_d=0, max_d=3, start_q=0, max_q=5, seasonal=True, random_state=42,
stepwise=True, suppress_warnings=True, error_action='ignore', trace=True)

# %%
# fit the model

pmarima.fit(train)

# %%
# get the best model

model = pmarima.model_
model.summary()

# %% [markdown]
# - the best model from the search is with p=1, d=1 and q=5

# %%
# get the predictions from the best model, with the confidence intervals

pm_test_pred, confs = model.predict(n_periods=len(test), return_conf_int=True)
pm_test_pred

# %%
# store the test results

pm_test_results = pd.DataFrame(columns=['test','pm_test_pred'])
pm_test_results['test'] = np.exp(test)
pm_test_results['pm_test_pred'] = np.exp(pm_test_pred)

pm_test_results

# %%
# plot the predictions from the best model and the confidence interval

plt.plot(np.exp(test), label='test')
plt.plot(np.exp(pm_test_pred), label='pred')
plt.fill_between(test.index, np.exp(confs[:,0]), np.exp(confs[:,1]), color='blue', alpha=0.1)
plt.legend()

# %%
# get the metrics for the best model

print('MSE', mean_squared_error(pm_test_results['test'], pm_test_results['pm_test_pred']))
print('RMSE', root_mean_squared_error(pm_test_results['test'], pm_test_results['pm_test_pred']))
print('MAPE', mean_absolute_percentage_error(pm_test_results['test'], pm_test_results['pm_test_pred']))

# %% [markdown]
# ### The RMSE of the best model, with p=1, d=1 and q=5, is 88.06573335360618, which is higher the RMSE of the ARIMA with p=3, d=2, q=3

# %% [markdown]
# ### Use the Prophet model to predict

# %%
# import prophet

from prophet import Prophet

# %%
# create the model

m = Prophet()
prophet_train = np.exp(train).reset_index()
prophet_train

# %%
# rename the columns to ds and y

prophet_train.columns = ['ds','y']
prophet_train

# %%
# fit the data

m.fit(prophet_train)

# %%
# get the predictions
# make a future dataframe
future = m.make_future_dataframe(periods=len(test), freq='ME')
forecast = m.predict(future)

forecast

# %%
# plot the test and predictions

m.plot(forecast)

plt.xlim(test.index[0], test.index[-1])
plt.plot(np.exp(test), label='test', color='red')

# %%
# save the predictions for only the test time period

m_test_preds = [forecast.loc[forecast['ds'] == month, 'yhat'].values[0] for month in test.index]
m_test_preds

# %%
# save the predictions in a dataframe

m_test_results = pd.DataFrame(columns = ['test', 'm_test_pred'])
m_test_results['test'] = np.exp(test)
m_test_results['m_test_pred'] = m_test_preds

m_test_results

# %%
# get the metrics for the Prophet model

print('MSE', mean_squared_error(m_test_results['test'], m_test_results['m_test_pred']))
print('RMSE', root_mean_squared_error(m_test_results['test'], m_test_results['m_test_pred']))
print('MAPE', mean_absolute_percentage_error(m_test_results['test'], m_test_results['m_test_pred']))

# %% [markdown]
# ### The RMSE for the Prophet model is 43.5315613273586, which is much lower than both the ARIMA and the best model from the PMDARIMA search.

# %% [markdown]
# ### Remove Seasonality using Seasonal from Seasonal Decompose

# %%
# imports

from statsmodels.tsa.seasonal import seasonal_decompose

# %%
# get the results

seasonal_results = seasonal_decompose(df_log['#Passengers'].dropna())
seasonal_results.plot()

# %%
# get the seasonal from the results

seasonal = seasonal_results.seasonal
seasonal

# %%
# remove seasonality

df_deseasoned = (df_log['#Passengers'] - seasonal)
df_deseasoned.dropna(inplace=True)
np.exp(df_deseasoned)

# %%
# visualized the deseasoned plot

df_deseasoned.plot()

# %%
# use the deseasoned data to fit the ARIMA model
# first, check the acf and pacf plots

plot_acf(df_deseasoned.diff().diff().dropna())

# %% [markdown]
# - best q = 2

# %%
# check the pacf plot

plot_pacf(df_deseasoned.diff().diff().dropna())

# %% [markdown]
# - best p = 3

# %% [markdown]
# ### best ARIMA parameters based on acf and pacf plots are p=3, d=2, q=2

# %%
# train test split

train_ds, test_ds = train_test_split(df_deseasoned, test_size=12, shuffle=False)

display(train.tail())
display(test.head())

# %%
# fit the ARIMA model and fit

q=2
d=2
p=3

model3_2_2 = ARIMA(train_ds, order=(p, d, q))
results = model3_2_2.fit()

results.summary()

# %% [markdown]
# - the AIC, BIC and HQIC are all much lower than the first ARIMA model

# %%
# get the predictions

train_ds_preds = results.fittedvalues

results_ds = pd.DataFrame(columns = ['train_ds', 'train_ds_preds'])
results_ds['train_ds'] = np.exp(train_ds)
results_ds['train_ds_preds'] = np.exp(train_ds_preds)
results_ds.plot(alpha=0.5)
plt.xlim(train.index[-30], train.index[-1])

# %%
# forecast the test

forecast_ds_12mo = results.forecast(steps=len(test_ds))
forecast_ds_12mo

# %%
# store the test results

test_ds_results = pd.DataFrame(columns=['test_ds', 'test_ds_pred'])
test_ds_results['test_ds'] = np.exp(test_ds)
test_ds_results['test_ds_pred'] = np.exp(forecast_ds_12mo)

test_ds_results

# %%
# plot the results
test_ds_results.plot()

# %%
# get the metrics for deseasonalized data using the ARIMA model

print('MSE', mean_squared_error(test_ds_results['test_ds'], test_ds_results['test_ds_pred']))
print('RMSE', root_mean_squared_error(test_ds_results['test_ds'], test_ds_results['test_ds_pred']))
print('MAPE', mean_absolute_percentage_error(test_ds_results['test_ds'], test_ds_results['test_ds_pred']))

# %% [markdown]
# ### The RMSE for the deseasonalized data using the ARIMA model with q=2, d=2, p=3 is 19.809313458020338. This is the best model so far. Note that the predictions are still deseasonalized, the seasonality needs to be added back.

# %%
# add back seasonality

restored_preds = forecast_ds_12mo + seasonal.loc[forecast_ds_12mo.index]
restored_preds

# %%
# store the restored results

test_restored_results = pd.DataFrame(columns=['test', 'restored_preds'])
test_restored_results['test'] = np.exp(test)
test_restored_results['restored_preds'] = np.exp(restored_preds)

test_restored_results

# %%
# get the metrics for restored data using the ARIMA model

print('MSE', mean_squared_error(test_restored_results['test'], test_restored_results['restored_preds']))
print('RMSE', root_mean_squared_error(test_restored_results['test'], test_restored_results['restored_preds']))
print('MAPE', mean_absolute_percentage_error(test_restored_results['test'], test_restored_results['restored_preds']))

# %% [markdown]
# ### The RMSE for the deseasonalized and reseasonalized data using the ARIMA model with q=2, d=2, p=3 is 20.117920293574056. This is the best model so far.

# %%
