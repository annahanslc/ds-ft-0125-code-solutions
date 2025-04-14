# %% [markdown]
# # 3. Collaborative Filtering - User Based

# %%
# imports

import pandas as pd
import numpy as np
import json
import gzip

from collections import defaultdict

# %%
# define file paths

path_meta = 'original_data/meta-Utah.json.gz'
path_pivot_subset = 'data/subset.parquet'
path_reviews_subset = 'data/subset_reviews.parquet'
path_user_sim = 'data/subset_user_sim.parquet'
path_item_sim = 'data/subset_item_sim.parquet'

# %%
# import data

user_sim = pd.read_parquet(path_user_sim)

# %%
# define a function for reading the data using a generator

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# %%
# import reviews subset

reviews = pd.read_parquet(path_reviews_subset)
reviews

# %%
# define a function to get all the reviews for a user, sorted by rating in descending order

def get_user_rated_sorted(user_id, df):
    user_reviews = df[df['user_id'] == user_id]
    user_rated = dict(zip(user_reviews['gmap_id'], user_reviews['rating']))
    user_rated = pd.DataFrame(list(user_rated.items()), columns=['gmap_id', 'rating'])
    user_rated.sort_values(by='rating', ascending=False, inplace=True)
    return user_rated

# %%
# test the function

get_user_rated_sorted('100518858506638839555', reviews)

# %%
user_rated_sorted = get_user_rated_sorted('100518858506638839555', reviews)
user_rated_sorted[user_rated_sorted['rating'] == 5]

# %%
# define a function to get a user's favorite places, favorites defined as being rated 4 or higher

def get_favorites(user_id, df):
  user_rated_sorted = get_user_rated_sorted(user_id, df)
  favorites = user_rated_sorted[user_rated_sorted['rating'] >= 4]
  return favorites

# %%
# test the function

favorite_test = get_favorites('111182595077674366891', reviews)
favorite_test

# %% [markdown]
# # User-based collaborative filtering

# %%
# get the id of a sample user

sample_user_id = user_sim.index[5]
sample_user_id

# %%
# get the list of what the user already rated

user_rated_sorted = get_user_rated_sorted(sample_user_id, reviews)
user_rated_sorted

# %%
# check the similarities for the user, sorted highest to lowest, index starting at 1 to remove itself

user_sim.loc[sample_user_id].sort_values(ascending=False)[1:]

# %%
# get the top 50 most similar users

similar_users_50 = user_sim.loc[sample_user_id].sort_values(ascending=False)[1:51]
similar_users_50

# %%
# use get_favorites function to get the favorites for the top 50 most similar users

from collections import defaultdict

favs_of_similar_users = defaultdict(float)

for key in similar_users_50.keys():
  favs = get_favorites(key, reviews)
  for _, row in favs.iterrows():
    if row['gmap_id'] not in set(user_rated_sorted['gmap_id']):
      favs_of_similar_users[row['gmap_id']] += row['rating']

favs_of_similar_users

# %%
# sort by the total rating

favs_of_similar_users_df = pd.DataFrame(list(favs_of_similar_users.items()), columns=['gmap_id','total_rating'])
favs_of_similar_users_df.sort_values(by='total_rating', ascending=False, inplace=True)
favs_of_similar_users_df

# %% [markdown]
# # Define helper functions

# %%
# function to get all the reviews for a user, sorted by rating in descending order

def get_user_rated_sorted(user_id, df):
    user_reviews = df[df['user_id'] == user_id]
    user_rated = dict(zip(user_reviews['gmap_id'], user_reviews['rating']))
    user_rated = pd.DataFrame(list(user_rated.items()), columns=['gmap_id', 'rating'])
    user_rated.sort_values(by='rating', ascending=False, inplace=True)
    return user_rated

# define a function to get a user's favorite places, favorites defined as being rated 4 or higher

def get_favorites(user_id, df):
  user_rated_sorted = get_user_rated_sorted(user_id, df)
  favorites = user_rated_sorted[user_rated_sorted['rating'] >= 4]
  return favorites

# create a function to return the business name using the gmap_id

def get_business_name(gmap_id):
  meta_generator = parse(path_meta)
  for place in meta_generator:
    if place.get('gmap_id') == gmap_id:
      name = place['name']
      break
  return name

# define a function to get n most popular businesses, popular determined as 1) the most # of reviews and 2) highest average review

def get_popular(n, df):
  popular = df.groupby('gmap_id')['rating'].agg(['count','mean']).sort_values(by=['count','mean'], ascending=False)
  return popular.head(n)

# %% [markdown]
# # Create a user-based recommender function

# %%
# create a function to recommend n number of businesses to a user based on what similar users like

def user_based_recommendations(user_id, df, user_sim, n_recs):

  # get the list of what the user has already rated
  user_rated_sorted = get_user_rated_sorted(user_id, df)

  # change to a set for faster lookup later
  set_user_rated_sorted = set(user_rated_sorted['gmap_id'])

  # if the user_id is not in predicted_df
  if user_id in user_sim.keys():
    print(f"{user_id} found in dataset")

    # get the top 50 most similar users
    similar_users_50 = user_sim.loc[user_id].sort_values(ascending=False)[1:51]

    # create a dictionary to store the similar users' favorites
    favs_of_similar_users = defaultdict(float)

    # create a loop to iterate through the 50 users and get their favorite businesses
    for key in similar_users_50.keys():
      favs = get_favorites(key, df)
      for _, row in favs.iterrows():
        if row['gmap_id'] not in set_user_rated_sorted:
          favs_of_similar_users[row['gmap_id']] += row['rating']

    # sort by the total rating
    favs_of_similar_users_df = pd.DataFrame(list(favs_of_similar_users.items()), columns=['gmap_id','total_rating'])
    favs_of_similar_users_df.sort_values(by='total_rating', ascending=False, inplace=True)
    favs_of_similar_users_df.reset_index(drop=True, inplace=True)

    # get n recommendations
    n_favs = favs_of_similar_users_df.head(n_recs).copy()

    # add the business names
    n_favs['name'] = n_favs['gmap_id'].apply(get_business_name)

    return n_favs

  else:
    print(f"{user_id} not found — using popularity")
    popular = get_popular(n_recs, df)

    # name the columns and then resent index
    popular.columns = ['similarity','avg_rating']
    popular.reset_index(inplace=True)

    # add the business name to the df
    popular['name'] = popular['gmap_id'].apply(get_business_name)

    # reorder the columns to match the recs df
    popular = popular[['gmap_id','similarity','name','avg_rating']]

    return popular


# %%
# check the function 1

user_based_recommendations('116427980967433332299', reviews, user_sim, 10)

# %%
# check the function 2

user_based_recommendations('108160460172023739763', reviews, user_sim, 10)

# %%
# check the function 2

user_based_recommendations('12345', reviews, user_sim, 10)

# %%
