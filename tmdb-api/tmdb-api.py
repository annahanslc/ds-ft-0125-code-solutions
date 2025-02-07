# %% [markdown]
# # tmdb-api
#
# Exercise
# Generate an API key and an account on The Movie Database website.
#
# Make a request to the API and query for all Star Wars Movies.
#
# Format the json object into a DataFrame.
#
# Sort the movies in the DataFrame by popularity (highest to lowest).

# %% [markdown]
# ### Step 1: Retrieve all Star Wars movies from the api and transform into DataFrame format

# %%
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("IMDB_TOKEN")

search_url = "https://api.themoviedb.org/3/search/movie?query="
query = "Star+Wars"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(search_url+query, headers=headers)
result = response.json()['results']
df_result = pd.DataFrame(result)
df_result


# %% [markdown]
# ### Step 2: Sort the Star Wars movies by popularity in descending order

# %%
df_sorted = df_result.sort_values(by="popularity", ascending=False)
df_sorted

# %%
