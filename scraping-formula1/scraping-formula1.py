# %% [markdown]
# # scraping-formula1
#
# Practice web scraping using the following url: https://www.formula1.com/en/results/2022/drivers.
#
# Exercise
# Connect to the web server and make sure you receive an ‘ok’ status code.
#
# Scrape the page for the statistics stored in the table.
#
# Save the table as a data frame.
#
# Save the data frame as a .csv file, and re-read the file to make sure it saved correctly.
#
# Scrape the page using the pd.read_html( ) method.

# %%
from bs4 import BeautifulSoup as soup
import pandas as pd
import requests

# %% [markdown]
# ##### Step 1: Connect to the webserver and get an 'ok' status

# %%
url = "https://www.formula1.com/en/results/2022/drivers"
response = requests.get(url)
response.status_code

# %% [markdown]
# ##### Step 2: Scrape the page for statistics stored in the table

# %%
res_soup = soup(response.content)
res_soup

# %%
res_table = res_soup.find('table')

# %% [markdown]
# ##### Step 3: Saving the table as a dataframe

# %%
from io import StringIO

df_res = pd.read_html(StringIO(str(res_table)))[0]
df_res

# %% [markdown]
# ##### Step 4: Save the dataframe as a .csv file, and re-read it to double check

# %%
df_res.to_csv('formula1_2022.csv',index=False)

# %%
new_df_res = pd.read_csv('formula1_2022.csv')
new_df_res

# %% [markdown]
# ##### Step 5: Scrape the page using the pd.read_html() method

# %%
new_scrape = pd.read_html('https://www.formula1.com/en/results/2022/drivers')
new_scrape[0]

# %%
