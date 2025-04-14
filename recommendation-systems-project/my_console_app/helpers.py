
import pandas as pd
import json
import gzip

# define file paths

path_meta = '../original_data/meta-Utah.json.gz'
path_pivot_subset = '../data/subset.parquet'
path_reviews_subset = '../data/subset_reviews.parquet'
path_user_sim = '../data/subset_user_sim.parquet'
path_item_sim = '../data/subset_item_sim.parquet'

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
