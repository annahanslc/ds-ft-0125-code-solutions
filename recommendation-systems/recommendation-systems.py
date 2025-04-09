# %% [markdown]
# # Recommendation Systems

# %% [markdown]
# 1. Write a function that recommends 5 beauty products for each user based on popularity among other users.
#
# 2. Write a function that recommends 5 beauty products for each user based on next items purchased by other users.

# %%
# imports

import pandas as pd
import numpy as np

# %%
# get data

beauty_original = pd.read_json('Beauty.json', lines=True)
beauty_original

# %% [markdown]
# - overall is the rating
# - asin is the item number
# - date and review time are the same, and unixreviewtime
# - reviewerID is the user info
#
# I will use:
# - asin, unixreviewtime, reviewerID, and overall

# %%
# save the asin, unixreviewtime, reviewerID and overall to a dataframe

beauty = beauty_original[['asin', 'unixReviewTime', 'reviewerID', 'overall']].copy()

# %%
# check the dataframe

beauty

# %% [markdown]
# ### 1. Write a function that recommends 5 beauty products for each user based on popularity among other users.

# %%
# groupby the product (asin) and get the sum of all scores to determine popularity
# then sort by the highest overall sum, and then by the highest overall count

asin_total_score = beauty.groupby('asin').agg(overall_sum=('overall','sum'), overall_count=('overall','count'))
asin_total_score.sort_values(by=['overall_sum','overall_count'], ascending=False, inplace=True)
asin_total_score

# %%
# write a function to get 5 product recommendations based on popularity among other users

def get_popular(df, n_recs) -> pd.Series:
  """
  Accepts a dataframe and the number of recommendations to return a series with those top recommendations
  """
  asin_total_score = df.groupby('asin').agg(overall_sum=('overall','sum'),
                                            count_next_rec=('overall','count'),
                                            avg_rating=('overall','mean'))
  asin_total_score.sort_values(by=['overall_sum','count_next_rec'], ascending=False, inplace=True)
  asin_total_score = asin_total_score
  recs = asin_total_score.head(n_recs)[['count_next_rec','avg_rating']]

  return recs

# %%
# test the function

get_popular(beauty, 5)

# %% [markdown]
# ### 2. Write a function that recommends 5 beauty products for each user based on next items purchased by other users.

# %%
# create a new column to save the next times purchased, fill with nan for now

beauty['next_rec'] = np.nan
beauty

# %%
# sort the dataframe by the user (reviewerID), then by the review time (unixReviewTime) in ascending order

beauty.sort_values(by=['reviewerID','unixReviewTime'],ascending=True,inplace=True)
beauty

# %%
# group by user, then fill the next_rec with the pervious item purchased, so the last item will be NaN

beauty['next_rec'] = beauty.groupby('reviewerID')['asin'].shift(-1)

beauty.head(30)

# %%
# groupby the item (asin) to get the next_recs for each item

beauty[beauty['asin']=='B00021DJ32'].groupby('next_rec').agg(count_next_rec=('next_rec','count'),avg_rating=('overall', 'mean'))

# %%
# get all the reviews for a user

user_reviews = beauty[beauty['reviewerID'] == 'A3CIUOJXQ5VDQ2']
user_reviews

# %%
# get a list of items that the user already purchased to exclude them from the recommendations list

already_reviewed = user_reviews['asin'].unique()
already_reviewed

# %%
# get the most recent item that the user reviewed

most_recent = user_reviews[user_reviews['unixReviewTime'] == user_reviews['unixReviewTime'].max()]['asin']
most_recent.iloc[0]

# %%
# Write a function that recommends 5 beauty products for each user based on next items purchased by other users.

def rec_for_user(df, reviewerID, n_recs):
  """
  Accepts a dataframe, the user's reviewerID number, and the number of recommendations.
  Returns a dataframe with the item's asin as the index and number of reviews, and the average rating
  """

  # if new user, recommend popular items
  if reviewerID not in df['reviewerID'].values:
    recs = get_popular(df, n_recs)

  else:
    # get what the user has already reviewed
    user_reviews = df[df['reviewerID'] == reviewerID]

    # get the unique asin for those items
    already_reviewed = user_reviews['asin'].unique()

    # get the most recent item that the user reviewed
    most_recent = (user_reviews[user_reviews['unixReviewTime'] == user_reviews['unixReviewTime'].max()]['asin']).iloc[0]

    # filter out products that the user has already reviewed
    new_df = df[~df['next_rec'].isin(already_reviewed)].copy()

    # groupby the item (asin) to get the next_recs for each item
    recs = new_df[new_df['asin']==most_recent].groupby('next_rec').agg(count_next_rec=('next_rec','count'),avg_rating=('overall', 'mean'))

    # return only the number of recommendations desired
    recs = recs.head(n_recs)

  return recs


# %%
# test the function 1

rec_for_user(beauty, 'AZJMUP77WBQZQ', 5)

# %%
# test the function 2

rec_for_user(beauty, 'A10P0NAKKRYKTZ', 5)

# %%
# test the function 3 (new user)

rec_for_user(beauty, '12345', 5)
