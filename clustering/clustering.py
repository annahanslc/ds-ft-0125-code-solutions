# %% [markdown]
# # Clustering Assignment
#
# - skip the boundaries
# - nearest neighbor one is bonus

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn import set_config
set_config(transform_output='pandas')

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ### 1. Import Data

# %%
# import data and name the columns according to imports-85.names

df = pd.read_csv('imports-85.data', header=None, names=['symboling','normalized-losses','make','fuel-type','aspiration','num_of_doors',
                                                        'body_style','drive_wheels','engine_location','wheel_base','length','width','height',
                                                        'curb-weight','engine-type','num_of_cylinders','engine_size','fuel_system','bore','stroke',
                                                        'compression_ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price'])
df.head()

# %% [markdown]
# ### 2. EDA

# %%
# check the 2 columns we will be using for nulls

df[['price','horsepower']].info()

# %% [markdown]
# - no nulls, but type is 'object', check values next

# %%
# check values of price

df['price'].value_counts()

# %%
# check the observations that have "?" in price

df[df['price']=='?']

# %% [markdown]
# - there are 4 "?" in price feature.
# - in the data dictionary, '?' are placeholders for missing values.
# - looking at the observations, I cannot see any particular reason for why the value is missing, so they are missing at random
# - since we will only be using 2 features, to impute either one of the features would be misleading the relationship between the 2 features
# - there are only 2 observations out of 205, so the amount of information lost is not significant
# - for the above reasons, I will drop the observations from the dataset.

# %%
# check values of horsepower

df['horsepower'].value_counts()

# %%
# check the observations that have "?" in price

df[df['horsepower']=='?']

# %% [markdown]
# - there are 2 "?" in price feature.
# - in the data dictionary, '?' are placeholders for missing values.
# - looking at the observations, I cannot see any particular reason for why the value is missing, so they are missing at random
# - since we will only be using 2 features, to impute either one of the features would be misleading the relationship between the 2 features
# - there are only 2 observations out of 205, so the amount of information lost is not significant
# - for the above reasons, I will drop the observations from the dataset.

# %%
# drop observations container "?" in either feature from dataset

df.drop(df[(df['price']=='?') | (df['horsepower']=='?')].index, inplace=True)
df[['price','horsepower']].info()

# %%
# check for duplicates, each row is 1 car, so checking for duplicated cars

df.duplicated().sum()

# %% [markdown]
# - no duplicates

# %%
# choose the features we would like to model with: price and horsepower

X = df[['price', 'horsepower']].copy()
X.head()

# %%
# scale the data

X_scaled = StandardScaler().fit_transform(X)
X_scaled.head()

# %% [markdown]
# # Exercises

# %% [markdown]
# ### 1. Using 'price' and 'horsepower' columns from import-85.data, generate a KMeans model.

# %%
# define the model and fit

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# %% [markdown]
# 1.1.a. Find the optimal value of k using **elbow method**, silhouette score, and calinski-harabasz score.

# %%
# create a range to try different k values
k_range = range(2,15)

# create a list to store the inertias
inertias = []

# loop over the different k's and calculate the corresponding inertia score and add to list
for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_scaled)
  inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias)
plt.axvline(x=4, color='red')
plt.axvline(x=7, color='red');

# %% [markdown]
# - The elbow appears between k=4 and k=7

# %% [markdown]
# 1.1.b. Find the optimal value of k using elbow method, **silhouette score**, and calinski-harabasz score.

# %%
# create a list to store the silhouette scores
silhouette_scores = []

# loop over the different k's and calculate the corresponding inertia score and add to list
for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_scaled)
  silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.plot(k_range, silhouette_scores)
plt.axvline(x=2, color='r')
plt.axvline(x=4, color='r');

# %% [markdown]
# - the best silhouette scores are when k is the smallest, this suggests that the best natural clustering is in a small number of groups and these groups are well-separated.
# - the best k based on the silhouette score would be between k=2 and k=4

# %% [markdown]
# 1.1.c. Find the optimal value of k using elbow method, silhouette score, and **calinski-harabasz** score.

# %%
# create a list to store the ch scores
ch_scores = []

# loop over the different k's and calculate the corresponding inertia score and add to list
for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_scaled)
  ch_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))

plt.plot(k_range, ch_scores)
plt.axhline(y=433, color='g')
plt.axvline(x=5, color='r')
plt.axvline(x=7, color='r')
plt.axvline(x=14, color='r');

# %% [markdown]
# - The best k is where the CH score is the highest, which occurred at 5, 7 and 14.
# - Based on the results from elbow and silhouette, I will not increase k range to extend beyond 14.

# %% [markdown]
# ### Best k
#
# - Based on the results of the elbow method, silhouette score, and the calinski-harabasz score, the best overall k is 4

# %% [markdown]
# ### 1.2. Plot clusters with cluster centers.

# %%
# retrain the model with the best k, which is k=4

kmeans4 = KMeans(n_clusters=4, random_state=42)
kmeans4.fit_transform(X_scaled)

# %%
# add the cluster labels as a few column in the df

X_vis = X_scaled.copy()
X_vis['cluster'] = kmeans4.labels_
X_vis.head()

# %%
# plot the clusters

plt.scatter(X_vis['price'], X_vis['horsepower'], c=X_vis['cluster'])
plt.title('Price v Horsepower k=4')
plt.xlabel('price (scaled)')
plt.ylabel('horsepower (scaled)');

# %%
# add the cluster centers to the dataframe in order to plot them together

centers = pd.DataFrame(kmeans4.cluster_centers_)
centers['cluster'] = 5
centers = centers.rename(columns={0:'price',1:'horsepower'})
X_center = pd.concat([X_vis, centers])
X_center

# %%
# plot the clusters with centers

plt.scatter(X_center['price'], X_center['horsepower'], c=X_center['cluster'])
plt.title('Price v Horsepower k=4')
plt.xlabel('price (scaled)')
plt.ylabel('horsepower (scaled)');

# %% [markdown]
# - yellow points denote the cluster centers

# %% [markdown]
# ### 2. Using 'price' and 'horsepower' columns from import-85.data, generate a DBSCAN model.

# %%
# set a starting min_samples

min_samples = 4

# %%
# use NearestNeighbors to find the best epsilon

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

distances

# %%
# get only the furthest one out

n_th_neighbor = distances[:, min_samples -1]

# sort
n_th_neighbor_sorted = np.sort(n_th_neighbor)
n_th_neighbor_sorted

# %%
# plot the sorted neighbors

plt.plot(n_th_neighbor_sorted)
plt.ylabel(f'distance to {min_samples}th neighbor')
plt.ylabel('sorted min_samples neighbor distance')
plt.axhline(y=0.2, color='r')
plt.grid()

# %%
# set the epsilon

epsilon = 0.4

# %%
# define the model

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X_scaled)

# %%
# check all the unique values to see how many clusters were made

np.unique(dbscan.labels_)

# %%
# plot the points

plt.scatter(X_scaled.iloc[:,0], X_scaled.iloc[:,1], c=dbscan.labels_)
plt.title('Price v Horsepower DBSCAN min_samples=4')
plt.xlabel('price (scaled)')
plt.ylabel('horsepower (scaled)');

# %% [markdown]
# - the expensive and high horsepower cars were all determined to be anomalies by the model when min_samples=4
# - this does not seem to be the best clustering, I will try to tune the model

# %%
# create a function to quickly tune and plot DBSCAN

def tune_DBSCAN(min_samples=4, epsilon=0.4):
  dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X_scaled)
  plt.scatter(X_scaled.iloc[:,0], X_scaled.iloc[:,1], c=dbscan.labels_)
  plt.title('Price v Horsepower DBSCAN min_samples=4')
  plt.xlabel('price (scaled)')
  plt.ylabel('horsepower (scaled)')
  plt.show()

# %% [markdown]
# - tried a different combinations of min_samples and epsilon

# %%
# best model

tune_DBSCAN(3, 0.7)

# %% [markdown]
# - DBSCAN with min_sample of 3 and epsilon of 0.7 clusters the data points into 2 groups that have a gap between them
# - there are 2 anomalies that are further away from the clusters, visually these are reasonable to consider as anomalies
# - this model clusters the data points in a reasonable way

# %% [markdown]
# ### 3. Using mlb_batting_cleaned.csv, write a function that takes a player's name and shows the 2 closest players using the nearest neighbors algorithm.

# %%
# import data

mlb = pd.read_csv('mlb_batting_cleaned.csv')
mlb.head()

# %% [markdown]
# ### EDA

# %%
# check for null and data type

mlb.info()

# %% [markdown]
# - no nulls
# - need to one hot encode Tm and Lg
# - scale all the numeric features
# - drop the name

# %%
# preprocess data

ohe_pipe = Pipeline([
  ('ohe', OneHotEncoder(sparse_output=False))
])

num_pipe = Pipeline([
  ('scaler', StandardScaler())
])

num_cols = mlb.select_dtypes(include='number').columns

preprocessor = ColumnTransformer([
  ('ohe_pipe', ohe_pipe, ['Tm','Lg']),
  ('num_pipe', num_pipe, num_cols),
  ], remainder='drop', verbose_feature_names_out=False)

mlb_proc = preprocessor.fit_transform(mlb)
mlb_proc.head()

# %%
# define the model

min_players = 3
nearest = NearestNeighbors(n_neighbors=min_players)
nearest.fit(mlb_proc)

# %%
# define a function

def get_similar_to():
  name = input('Input player name:')
  idx = mlb[mlb['Name'] == name].index[0]
  distances, indices = nearest.kneighbors(mlb_proc.iloc[[idx]])
  first_idx = indices[0][1]
  first_neighbor = mlb['Name'].iloc[first_idx]
  second_idx = indices[0][2]
  second_neighbor = mlb['Name'].iloc[second_idx]

  print(f'Input player name: {name}')
  print(f'The first closest player: {first_neighbor}')
  print(f'The second closest player: {second_neighbor}')

# %%
get_similar_to()
