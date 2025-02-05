# %% [markdown]
# # Practicing reading data

# %%
import pandas as pd

# %% [markdown]
# ### Part 1: mtcars.csv

# %% [markdown]
# ##### Step 1: Reading the data from file into a DataFrame

# %%
mtcars = pd.read_csv('data/mtcars.csv')
mtcars.head(10)

# %% [markdown]
# ##### Step 2: Save the data back into a file with a different name

# %%
mtcars.to_csv('data/mtcars_new.csv', index=False)

# %% [markdown]
# ##### Step 3: Checking if the file saved properly

# %%
mtcars_new = pd.read_csv('data/mtcars_new.csv')
mtcars_new.head(10)

# %% [markdown]
# ### Part 2: beer.txt

# %% [markdown]
# ##### Step 1: Reading the file

# %%
beer_original = pd.read_csv('data/beer.txt', delim_whitespace=True)
beer_original.head(10)

# %% [markdown]
# ##### Step 2: Saving the file back into a txt with a different name but same format

# %%
beer_original.to_csv('data/beer_new.txt', sep='\t', index=False)

# %% [markdown]
# ##### Step 3: Checking if the file saved properly

# %%
new_beer = pd.read_csv('data/beer_new.txt', delim_whitespace=True)
new_beer.head(10)

# %% [markdown]
# ### Part 3: NHL 2015-16.xlsx

# %% [markdown]
# Step 1: Reading the file

# %%
nhl_original = pd.read_excel('data/NHL 2015-16.xlsx')
nhl_original.head(10)

# %% [markdown]
# ##### Step 2: Saving a new file

# %%
nhl_original.to_excel('data/nhl_new.xlsx', index=False)

# %% [markdown]
# ##### Step 3: Checking the new file

# %%
new_nhl = pd.read_excel('data/nhl_new.xlsx')
new_nhl.head(10)

# %% [markdown]
# ### Part 4: Write a function that would read all the previous file types without exceptions

# %%
import os

def reading_data(filepath):
  try:
    if os.path.splitext(filepath)[1] == '.csv':
      return pd.read_csv(filepath)
    elif os.path.splitext(filepath)[1] == '.xlsx':
      return pd.read_excel(filepath)
    elif os.path.splitext(filepath)[1] == '.txt':
      return pd.read_csv(filepath, sep='\s+')
  except:
    print("Not supported")
    return None

# %% [markdown]
#

# %% [markdown]
# ##### Testing function on a csv

# %%
read_csv = reading_data('data/mtcars.csv')
read_csv.head(10)

# %% [markdown]
# ##### Testing function on a txt

# %%
read_txt = reading_data('data/beer.txt')
read_txt.head(10)

# %%
read_excel = reading_data('data/NHL 2015-16.xlsx')
read_excel.head(10)

# %% [markdown]
# ##### Testing function on an invalid file type

# %%
read_na = reading_data("data/testdata.abc")
print(read_na)

# %%
