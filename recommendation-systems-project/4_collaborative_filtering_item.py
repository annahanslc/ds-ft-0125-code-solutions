# %% [markdown]
# # 4. Collaborative Filtering - Item Based

# %%
# imports

import pandas as pd
import numpy as np
import json
import gzip

# %%
# define file paths

path_meta = 'original_data/meta-Utah.json.gz'
path_pivot_subset = 'data/subset.parquet'
path_reviews_subset = 'data/subset_reviews.parquet'
path_user_sim = 'data/subset_user_sim.parquet'
path_item_sim = 'data/subset_item_sim.parquet'

# %%
# import the item similarities

item_sim = pd.read_parquet(path_item_sim)

# %%
# import the reviews for the 10,000 subset of users

reviews = pd.read_parquet(path_reviews_subset)

# %% [markdown]
# ### Helper functions

# %%
# define a function for reading the data using a generator

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# define a function to get all the reviews for a user, sorted by rating in descending order

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
# ### Recommender function

# %%
def item_based_recommendations(user_id, df, item_sim, n_recs):
  """
  Accepts the following parameters and returns a DataFrame containing n_recs recommendations.

  user_id = user_id
  df = the pivoted table
  item_sim = the item similarities matrix
  n_recs = the number of recommendations desired
  """

  # get the list of what the user already rated
  user_rated_sorted = get_user_rated_sorted(user_id, df)

  # get favorites for user
  user_favs = get_favorites(user_id, df)

  # create an empty dictionary to add recommendations to
  recs_dict = {}

  # for each favorite place in the user's list of favorite place:
  for fav in user_favs.values:
    fav = fav[0]

    # get the similar places with cosine similarity > 0
    user_item_sims = item_sim.loc[fav].sort_values(ascending=False)[1:]
    similar_places = user_item_sims[user_item_sims.values > 0]

    # for each of the similar places:
    for index, value in similar_places.items():
      place = index

      # if they are not in the list of places the user has already rated
      if place not in user_rated_sorted and place not in recs_dict:

      # then add the place's gmap_id and the similarity, to the dictionary.
        recs_dict[place] = similar_places[place]

      # if the place is already in the dictionary, then add the cosine similarity to the existing similarity
      if place in recs_dict:
        recs_dict[place] += similar_places[place]

  # turn the dictionary into a dataframe
  recs_df = pd.DataFrame(list(recs_dict.items()), columns=['gmap_id','similarity'])

  # sort the dataframe by the total cosine similarity, from highest to lowest
  recs_df.sort_values(by='similarity', ascending=False, inplace=True)

  # narrow down the recommendation to n * 2 by using .head
  recs_df = recs_df.head(n_recs*2)

  # for each place, add the business name
  recs_df['name'] = recs_df['gmap_id'].apply(get_business_name)

  # for each place, add the average rating
  recs_df['avg_rating'] = recs_df['gmap_id'].apply(get_business_rating)

  # sort by 1) similarity, 2) avg_rating
  recs_df.sort_values(by=['similarity','avg_rating'], ascending=False, inplace=True)

  # remove businesses that have an average rating under 3.5 stars
  recs_df = recs_df[recs_df['avg_rating'] > 3.5]

  # limit the length using head of n_recs
  recs = recs_df.head(n_recs)

  # calculate the difference between the length of list of recommendation and n_recs
  diff = n_recs - recs['gmap_id'].count()

  # if the difference is 0, return the list of recommendations
  if diff == 0:
    return recs

  # else create an additional list of recommendations using get_popular, with n = the difference
  else:
    popular = get_popular(diff, df)

    # name the columns and then resent index
    popular.columns = ['similarity','avg_rating']
    popular.reset_index(inplace=True)

    # add the business name to the df
    popular['name'] = popular['gmap_id'].apply(get_business_name)

    # reorder the columns to match the recs df
    popular = popular[['gmap_id','similarity','name','avg_rating']]


  # add the additional list to the bottom of the original list of recommendations
    recs = pd.concat([recs, popular], ignore_index=True)

  return recs


# %%
# test the function 1

item_based_recommendations('104620742288190585924', reviews , item_sim, 10)

# %%
# test the function 2

item_based_recommendations('12345', reviews , item_sim, 10)

# %%
# test the function 3

item_based_recommendations('111717473911684632928', reviews , item_sim, 50)

# %%
