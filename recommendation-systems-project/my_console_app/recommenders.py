from helpers import *

import pandas as pd
from collections import defaultdict

# define file paths

path_meta = '../original_data/meta-Utah.json.gz'
path_pivot_subset = '../data/subset.parquet'
path_reviews_subset = '../data/subset_reviews.parquet'
path_user_sim = '../data/subset_user_sim.parquet'
path_item_sim = '../data/subset_item_sim.parquet'
path_svd_preds = '../data/svd_preds.parquet'


######### USER BASED RECOMMENDER FUNCTION #########

# create a function to recommend n number of businesses to a user based on what similar users like

def user_based_recommendations(user_id, n_recs) -> pd.DataFrame:
  """
  Accepts the following parameters and returns a DataFrame containing n_recs recommendations.
  This function uses user-based collaborative filtering to generate recommendations.s

  user_id = user_id
  n_recs = the number of recommendations desired
  """

  # get the user similarities

  user_sim = pd.read_parquet(path_user_sim)

  # get the reviews for the 10,000 subset of users

  df = pd.read_parquet(path_reviews_subset)

  # get the list of what the user has already rated
  user_rated_sorted = get_user_rated_sorted(user_id, df)

  # change to a set for faster lookup later
  set_user_rated_sorted = set(user_rated_sorted['gmap_id'])

  # if the user_id is in predicted_df
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




######### ITEM BASED RECOMMENDER FUNCTION #########

def item_based_recommendations(user_id, n_recs) -> pd.DataFrame:
  """
  Accepts the following parameters and returns a DataFrame containing n_recs recommendations.
  This function uses item-based collaborative filtering to generate recommendations.s

  user_id = user_id
  n_recs = the number of recommendations desired
  """
  # get the item similarities

  item_sim = pd.read_parquet(path_item_sim)

  # get the reviews for the 10,000 subset of users

  df = pd.read_parquet(path_reviews_subset)

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
  diff = n_recs - len(recs)

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
    # recs = pd.concat([recs, popular], ignore_index=True)
    recs = popular

  return recs


######### SVD RECOMMENDER FUNCTION #########

# define a function to get n_recs for a user using SVD model
# the recommendations are the items with the highest predicted ratings

def get_svd_recommendations(user_id, n_recs):
  org_df = pd.read_parquet(path_reviews_subset)
  pred_df = pd.read_parquet(path_svd_preds)

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
    print('user has reviews')
    # get the list of what the user already rated
    user_rated_sorted = get_user_rated_sorted(user_id, org_df)
    num_rated = len(user_rated_sorted)

    # create a dictionary to add recommendations to
    recs = {}

    # get all recommedations for this user, sorted in descending order of predicted rating
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
