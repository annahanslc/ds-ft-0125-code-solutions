# %% [markdown]
# # reading-json
#
# Exercise
# Read the json_file.txt file and print the values. The expected output should look like this: image
#
# Read the daily_covid_cases.json file and print the values. The expected output should look like this: image
#

# %% [markdown]
# ### Part 1: reading json_file.txt and printing them in the displayed format

# %%
import json
json_file_loaded = json.load(open('json_file.txt'))

for dict in json_file_loaded:
  for key, value in dict.items():
    print(key+":", value)
  print("="*17)

# %% [markdown]
# ### Part 2: reading the daily_covid_cases.json file and printing the values

# %%
covid_cases = json.load(open('daily_covid_cases.json'))
covid_cases

for key1 in covid_cases.keys():
  print(key1)
  for key, value in covid_cases[key1].items():
    print(key+":", value)
  print("\n")


# %%


# %%
