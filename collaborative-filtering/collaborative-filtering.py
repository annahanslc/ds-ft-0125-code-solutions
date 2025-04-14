# %% [markdown]
# # Collaborative Filtering Homework

# %% [markdown]
# 1. Write a function that takes user index and the dataset (data and item) and returns 5 book recommendations based on User based collaborative filtering
# 2. Write a function that takes user index and the dataset (data and item) and returns 5 book recommendations based on Item based collaborative filtering

# %%
# import

import pandas as pd
import numpy as np

# %% [markdown]
# ### EDA

# %%
# import data (book ratings)

book_ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin1')
book_ratings

# %%
# checks for duplicates in book ratings (same user reviewing the same book more than once)

book_ratings.duplicated(['User-ID','ISBN']).sum()

# %%
# import data (books)

books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin1', on_bad_lines='skip')
books

# %%
# checks for duplicated books based ISBN

books.duplicated(['ISBN']).sum()

# %%
# checks for duplicated books based on title

books.duplicated(['Book-Title']).sum()

# %%
# checks for duplicated books based on title & author

books.duplicated(['Book-Title','Book-Author']).sum()

# %%
# checks for duplicated books based on title & author & year of publication

books.duplicated(['Book-Title','Book-Author','Year-Of-Publication']).sum()

# %% [markdown]
# - books that have the same title and author will mostly likely have the same content, and should be similarly liked by readers regardless of the year of publication.
# - before merging the title of the book into the ratings dataframe, I will first combine the Book Title with the book Author
# - ratings for books with the same Title AND Author can be averaged together

# %%
# combine the Book Title with the Book Author

books['Title-Author'] = books['Book-Title'] + '-' + books['Book-Author']
books

# %%
# import data (users)

users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin1')
users

# %% [markdown]
# - this data will not be used

# %% [markdown]
# ### Merge Data

# %%
# merge the book's title-author into the ratings dataframe

merged = pd.merge(book_ratings, books, on='ISBN')
merged

# %%
# check the number of unique books (based on Title-Author)

merged['Title-Author'].nunique()

# %%
# in the merged data, determine the books that have fewer than 10 reviews and create a boolean mask

review_counts = merged.groupby('ISBN')['User-ID'].transform('count')
mask_10up = review_counts > 10

# %%
# using the boolean mask, filter out the books that have 10 or fewer reviews

merged_10up = merged[mask_10up]
merged_10up

# %%
# check the number of books after filtering out the ones with 10 or fewer reviews

merged_10up['Title-Author'].nunique()

# %%
# create a pivot table using the user-ID and the book Title-Author, and any duplicates will be aggregating using the mean of the rating

pivot = merged_10up.pivot_table(index='User-ID', columns='Title-Author', values='Book-Rating', aggfunc='mean')
pivot.head(50)

# %%
# store the book titles and user id's for putting back later

books_titles = pivot.columns
user_ids = pivot.index

# %%
# transform into sparse

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

pivot_sparse = csr_matrix(pivot.fillna(0))
pivot_sparse

# %%
# calculate the user similarity

user_sim = cosine_similarity(pivot_sparse)
user_sim

# %%
# transform to Dataframe

user_sim_df = pd.DataFrame(user_sim, index=user_ids, columns=user_ids)
user_sim_df

# %%
# calculate the item similarity

item_sim = cosine_similarity(pivot_sparse.T)
item_sim

# %%
# transform to Dataframe

item_sim_df = pd.DataFrame(item_sim, index=books_titles, columns=books_titles)
item_sim_df

# %% [markdown]
# ### 1. Write a function that takes user index and the dataset (data and item) and returns 5 book recommendations based on User based collaborative filtering

# %%
# create a function that uses user-based collaborative filtering to make book recommendations


def user_based_rec(df, user_id, num_recs, like_threshold) -> pd.Series :
  """
  Accepts the following parameters and returns a Series containing the title-author of the desired number of book recommendations

  df: a dataframe of correlations between users
  user_id: the user_id
  num_recs: the number of book recommendations desired
  like_threshold: on a scale of 0-10, above what rating can a book be determined a user's favorite
  """

  # determine the similar users
  similar_users = df[df.index == user_id].T.sort_values(by=user_id, ascending=False)

  # get what the user already rated
  curr_user_rated = merged_10up[merged_10up['User-ID']==user_id]

  # create a function to get a user's favorite books
  def get_favs(user_id, like_threshold):
    user_rated = merged_10up[merged_10up['User-ID']==user_id]
    user_liked = user_rated[user_rated['Book-Rating'] > like_threshold].sort_values(by='Book-Rating', ascending=False)
    return user_liked['Title-Author']

  book_recs = []
  i = 0

  # create a while loop that will only continue while
  # 1) the number of recommendations has not been met
  # 2) that there are still similar users left to reference check
  while len(book_recs) < num_recs and i < len(similar_users[1:]):

    # try each user in the list of similar users
    user = similar_users[1:].index[i]
    # get the favorites for the current similar user
    new_fav = get_favs(user, like_threshold).head(1)

    # if the current similar user doesn't have an empty list of favorites
    if not new_fav.empty:
      # then get the title-author for their favorite book
      new_fav_title_author = new_fav.values[0]

      if new_fav_title_author not in curr_user_rated['Title-Author'].values and new_fav_title_author not in book_recs:
        book_recs.append(new_fav_title_author)

    i += 1

  return pd.Series(book_recs)

# %%
# test the function 1

user_based_rec(user_sim_df, 200273, 5, 5)

# %%
# test the function 2

user_based_rec(user_sim_df, 165, 10, 8)

# %% [markdown]
# ### 2. Write a function that takes user index and the dataset (data and item) and returns 5 book recommendations based on Item based collaborative filtering

# %%
# write a function that uses item-based collaborative filtering to make recommendations

def item_based_rec(df, user_id, num_recs, like_threshold):
  """
  Accepts the following parameters and returns a Series containing the title-author of the desired number of book recommendations

  df: a dataframe of correlations between books
  user_id: the user_id
  num_recs: the number of book recommendations desired
  like_threshold: on a scale of 0-10, above what rating can a book be determined a user's favorite
  """

  # get what the user already rated
  curr_user_rated = merged_10up[merged_10up['User-ID']==user_id]

  # create a function to get a user's favorite books
  def get_favs(df, user_id, like_threshold):
    user_rated = df[df['User-ID']==user_id]
    user_liked = user_rated[user_rated['Book-Rating'] > like_threshold].sort_values(by='Book-Rating', ascending=False)
    return user_liked['Title-Author']

  # get the favorites for the current user
  user_favorites = get_favs(merged_10up, user_id, like_threshold)

  # if a user does not have any favorites yet, recommend the most popular books
  if user_favorites.empty:
    most_rated = merged_10up.groupby('Title-Author').agg(sum_rating =('Book-Rating','sum')).sort_values(by='sum_rating', ascending=False)
    book_recs = most_rated.iloc[0:num_recs].index.values

  # otherwise, recommend similar books
  else:
    book_recs = []
    # start a counter to keep track of which favorite book we are looking at
    i=0
    # start a counter to keep track of which ranking of similar book we are looking at (the most similar, the second most similar, etc.)
    j=1

    # define a look to keep looking until we reach the desired number of recommendations
    while len(book_recs) < num_recs:

      # if we have reached the end of the list of user's favorite books, go back to the first one and get the next most similar
      if i == len(user_favorites):
        i = 0
        j += 1

      # get the book title from the user's list of favorites
      book_name = user_favorites.values[i]

      # using the correlation dataframe, get the list of most similar books (except for itself)
      similar_book = pd.DataFrame(df[book_name]).sort_values(by=book_name, ascending=False)[1:]

      # the new book to consider
      new_book = similar_book.index[j]

      # if the new book was not already reviewed by the user, and is not already in the recommended list, then add to rec list
      if new_book not in curr_user_rated['Title-Author'].values and new_book not in book_recs:
        book_recs.append(new_book)

      # move on to the next favorite book in the user's list
      i += 1

  return pd.Series(book_recs)

# %%
# test the function 1

item_based_rec(item_sim_df, 8, 5, 1)

# %%
# test the function 1

item_based_rec(item_sim_df, 8, 5, 5)

# %% [markdown]
# ### 3. Create a wrapper function to allow user to decide between user or item based filtering

# %%
# define a function that allows the user to choose

def get_recs(type, user_id, num_recs, like_threshold) -> pd.Series :
  """
  Accepts the following parameters and returns a Series of the desired number of book recommendations

  type = select between 2 types of collaborative filtering, either "user-based" or "item-based"
  user_id = the user_id
  num_recs = the number of book recommendations desired
  like_threshold = on a scale of 0-10, above what rating can a book be determined a user's favorite
  """

  if type == 'user-based':
    recs = user_based_rec(user_sim_df, user_id, num_recs, like_threshold)
  elif type == 'item-based':
    recs = item_based_rec(item_sim_df, user_id, num_recs, like_threshold)
  else:
    return print('Please choose the type of collaborative filtering, either "user-based" or "item-based"')

  return recs

# %%
# function test 1

get_recs('user-based', 8, 10, 5)

# %%
# function test 2

get_recs('item-based', 8, 10, 5)

# %%
# function test 3

get_recs('item', 8, 10, 5)
