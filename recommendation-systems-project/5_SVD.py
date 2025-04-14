# %% [markdown]
# # 5. SVD

# %%
# imports

import pandas as pd
import numpy as np
import json
import gzip

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

from sklearn.metrics import root_mean_squared_error

# %%
# define file paths

path_reviews_subset = 'data/subset_reviews.parquet'
path_meta = 'original_data/meta-Utah.json.gz'

# %%
# define a function for reading the data using a generator

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# %%
# import the reviews filtered by the 10,001 users in the subset

reviews = pd.read_parquet(path_reviews_subset)
reviews

# %%
# train-test-split

train, test = train_test_split(reviews, test_size=0.2, random_state=42)
train.shape, test.shape

# %%
# create 2 dataframes, for train and test, where the rows and columns are the same in both dataframes
# the rows are all the unique users in reviews and the columns are all the unique items in reviews

train_ratings = pd.DataFrame(index=reviews.user_id.unique(), columns=reviews.gmap_id.unique())
test_ratings = train_ratings.copy()

# %%
# fill the train dataframe row by row by using a loop

for i in range(len(train)):
  train_ratings.loc[train['user_id'].iloc[i], train['gmap_id'].iloc[i]] = train['rating'].iloc[i]

# %%
# fill the test dataframe row by row by using a loop

for i in range(len(test)):
  test_ratings.loc[train['user_id'].iloc[i], train['gmap_id'].iloc[i]] = train['rating'].iloc[i]

# %%
# make sure train and test are the same size

train_ratings.shape, test_ratings.shape

# %% [markdown]
# - I do not want to fill in the NaN's will 0's, because it will bias the ratings. Instead, I will fill it with the mean.
# - Before filling with the mean, I will first de-mean it, which will make the mean 0. That way, filling it with 0 is the same as filling with the mean, which will also allow me to keep the data sparse. Sparse data will squeeze out the 0's and only keep the ones with values.
# - To de-mean, I will use the mean of the non-NaN values.

# %%
# calculate the mean of the non-NaN values
userMean = np.nanmean(train_ratings)
print(f'The mean is {userMean}')

# demean the train data
train_demeaned = train_ratings - userMean

# check to make sure the resulting mean is 0
np.nanmean(train_demeaned)

# %%
# fill the train NaN's with 0

train_demeaned = train_demeaned.fillna(0).astype(float)

# %%
# make the data sparse

train_sparse = csr_matrix(train_demeaned)

# %%
# matrix factorization using svds

U, sigma, Vt = svds(train_sparse, k=50)

# diagonalize the sigma

sigma = np.diag(sigma)

# %%
# get the predictions by taking 2 dot products in succession
# 1. dot product of U and sigma
# 2. dot product of the above and Vt

predicted_ratings = U @ sigma @ Vt

# %%
# add back the mean

predicted_ratings += userMean

# turn it into a dataframe

predicted_df = pd.DataFrame(predicted_ratings, columns=train_ratings.columns, index=train_ratings.index)

# %%
# use clip to clip anything lower than 1 to 1, and anything higher than 5 to 5

predicted_df = np.clip(predicted_df, 1, 5)

# %%
# check the predictions

predicted_df

# %% [markdown]
# ### Helper Functions

# %%
# define a function to get all the reviews for a user, sorted by rating in descending order

def get_user_rated_sorted(user_id, df):
    user_reviews = df[df['user_id'] == user_id]
    user_rated = dict(zip(user_reviews['gmap_id'], user_reviews['rating']))
    user_rated = pd.DataFrame(list(user_rated.items()), columns=['gmap_id', 'rating'])
    user_rated.sort_values(by='rating', ascending=False, inplace=True)
    return user_rated

# define a function to get n most popular businesses, popular determined as 1) the most # of reviews and 2) highest average review

def get_popular(n, df):
  popular = df.groupby('gmap_id')['rating'].agg(['count','mean']).sort_values(by=['count','mean'], ascending=False)
  return popular.head(n)

# create a function to return the business name using the gmap_id

def get_business_name(gmap_id):
  meta_generator = parse(path_meta)
  for place in meta_generator:
    if place.get('gmap_id') == gmap_id:
      name = place['name']
      break
  return name

# create a function to return the business rating using the gmap_id

def get_business_rating(gmap_id):
  meta_generator = parse(path_meta)
  for place in meta_generator:
    if place.get('gmap_id') == gmap_id:
      avg_rating = place['avg_rating']
      break
  return avg_rating

# %% [markdown]
# ### SVD recommender function

# %%
'116427980967433332299' in reviews['user_id'].values

# %%
# define a function to get n_recs for a user using SVD model
# the recommendations are the items with the highest predicted ratings

def get_svd_recommendations(user_id, org_df, pred_df, n_recs):

  print(org_df['user_id'].values)
  # if the user_id is not in predicted_df
  if user_id not in org_df['user_id'].values:
    print(f'User {user_id} has no reviews, recommendations are based on the most popular businesses')

    # return the most popular places
    recs = get_popular(n_recs, org_df)
    recs.reset_index(inplace=True)

    recs.drop(columns='count', inplace=True)
    recs.columns = ['gmap_id','pred_rating']

    recs['name'] = recs['gmap_id'].apply(get_business_name)
    recs['avg_rating'] = recs['gmap_id'].apply(get_business_rating)


  else:
    # get the list of what the user already rated
    user_rated_sorted = get_user_rated_sorted(user_id, org_df)
    num_rated = len(user_rated_sorted)

    # create a dictionary to add recommendations to
    recs = {}

    # get all recommedations, sorted in descending order of predicted rating
    all_recs = pred_df.loc[user_id].sort_values(ascending=False).head(n_recs + num_rated)

    # for each item in all_recs, check if the user has already rated it
    for key, value in all_recs.items():
      if key not in user_rated_sorted['gmap_id'].values:
        recs[key] = value
      if len(recs) == n_recs:
        break

    # change recs into a dataframe with column names
    recs = pd.DataFrame(list(recs.items()), columns=['gmap_id','pred_rating'])

    # add the business name and average rating into the dataframe
    recs['name'] = recs['gmap_id'].apply(get_business_name)
    recs['avg_rating'] = recs['gmap_id'].apply(get_business_rating)

  return recs


# %%
# test function 1 (user with no reviews)

get_svd_recommendations('12345', reviews, predicted_df, 5)

# %%
# test function 2 (user with reviews)

get_svd_recommendations('116427980967433332299', reviews, predicted_df, 10)

# %% [markdown]
# # Calculate the RMSE for the model

# %%
# change the dataframes to numpy, numpy will change the

pred_np = predicted_df.to_numpy()
train_np = train_ratings.to_numpy()
test_np = test_ratings.to_numpy()

# %%
# filter preds for non-nan values in train, filter train for non-nan values

train_pred_np = pred_np[train_ratings.notna()]
train_np = train_np[train_ratings.notna()]

# filter preds for non-nan values in test, filter test for non-nan values

test_pred_np = pred_np[test_ratings.notna()]
test_np = test_np[test_ratings.notna()]

# check the shape of train

train_pred_np.shape, train_np.shape

# %%
# check the shape of test

test_pred_np.shape, test_np.shape

# %%
# calculate the RMSE

root_mean_squared_error(train_np, train_pred_np), root_mean_squared_error(test_np, test_pred_np)

# %% [markdown]
# # Tune the SVD model

# %% [markdown]
# ### 1. Tune on train

# %%
# tune the SVD model on k to get the model that yields the lowest RMSE

# define a variable to keep track of the best rmse, start with the highest number possible
best_rmse = float('inf')

# define a variable to keep track of the best k
best_k = None

# define range of k's to try
k_range = [20, 50, 100, 150]

# create a loop to try
for k in k_range:
  U, sigma, Vt = svds(train_sparse, k=k)
  sigma = np.diag(sigma)
  pred = U @ sigma @ Vt

  # create a mask with the non-nan's in the pre-demeaned and sparsified dataframe
  mask = train_ratings.notna().values

  # calculate the RMSE
  rmse = root_mean_squared_error(train_sparse.toarray()[mask], pred[mask])

  # if the rmse is lower, replace the previous as the best rmse, and best k
  if rmse < best_rmse:
    best_rmse = rmse
    best_k = k


# %%
best_rmse, best_k

# %% [markdown]
# ### 2. Tune on test

# %%
# demean test using the mean that was calculated from train

test_demeaned = test_ratings - userMean

# fillna with 0

test_demeaned = test_demeaned.fillna(0).astype(float)

# make the data sparse

test_sparse = csr_matrix(test_demeaned)

# %%
# tune the SVD model on k to get the model that yields the lowest test RMSE

# define a variable to keep track of the best rmse, start with the highest number possible
best_rmse = float('inf')

# define a variable to keep track of the best k
best_k = None

# define range of k's to try
k_range = [20, 50, 100, 150]

# create a loop to try
for k in k_range:
  U, sigma, Vt = svds(test_sparse, k=k)
  sigma = np.diag(sigma)
  pred = U @ sigma @ Vt

  # create a mask with the non-nan's in the pre-demeaned and sparsified dataframe
  mask = test_ratings.notna().values

  # calculate the RMSE
  rmse = root_mean_squared_error(test_sparse.toarray()[mask], pred[mask])

  # if the rmse is lower, replace the previous as the best rmse, and best k
  if rmse < best_rmse:
    best_rmse = rmse
    best_k = k


# %%
best_rmse, best_k

# %% [markdown]
# ### Retrain model using the best k

# %%
# matrix factorization using svds

U, sigma, Vt = svds(train_sparse, k=150)

# diagonalize the sigma

sigma = np.diag(sigma)

# %%
# get the predictions by taking 2 dot products in succession
# 1. dot product of U and sigma
# 2. dot product of the above and Vt

predicted_ratings = U @ sigma @ Vt

# %%
# add back the mean

predicted_ratings += userMean

# turn it into a dataframe

predicted_df = pd.DataFrame(predicted_ratings, columns=train_ratings.columns, index=train_ratings.index)

# %%
# use clip to clip anything lower than 1 to 1, and anything higher than 5 to 5

predicted_df = np.clip(predicted_df, 1, 5)

# %%
# check the predictions

predicted_df

# %%
predicted_df

# %%
predicted_df.loc['116427980967433332299'].sort_values(ascending=False).head(10)

# %%
# save the predictions

predicted_df.to_parquet('data/svd_preds.parquet', index=True)

# %%
