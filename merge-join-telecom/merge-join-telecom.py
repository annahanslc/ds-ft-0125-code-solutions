# %% [markdown]
# # merge-join-telecom
#
# Exercise
#
# Merge Tables.
#
# Explain the difference between inner and outer merge.
#
# Explain the difference between merge and join.
#
# Divide dataset into two dataframes by row (separate 50-50)(even vs odd)
#
# Concat the two dataframes.
#
# Handle duplicates.

# %% [markdown]
# ### Part 1: Merge tables

# %% [markdown]
# ##### Step 1: Import the csv files

# %%
import pandas as pd

file_path = 'Telstra Competition Data/'

# %%
event_type = pd.read_csv(file_path + 'event_type.csv')
log_feature = pd.read_csv(file_path + 'log_feature.csv')
resource_type = pd.read_csv(file_path + 'resource_type.csv')
severity_type = pd.read_csv(file_path + 'severity_type.csv')
train = pd.read_csv(file_path + 'train.csv')


# %% [markdown]
# ##### Step 2: Check if they imported correctly and look for common column

# %%
event_type.head()

# %%
log_feature.head()

# %%
resource_type.head()


# %%
severity_type.head()

# %%
train.head()

# %% [markdown]
# ##### Step 3: Merge the tables

# %%
merged_dfs = event_type

dfs_to_merge = [log_feature, resource_type, severity_type, train]

for df in dfs_to_merge:
  merged_dfs = merged_dfs.merge(df, how='inner', on='id')

merged_dfs


# %% [markdown]
# ### Part 2: Explain the difference between inner and outer merge.

# %% [markdown]
# An inner merge will only merge the rows where the columns have non-null values in both dataset, while an outer merge will merge all rows from both datasets even if they are missing values in some columns.

# %% [markdown]
# ### Part 3: Explain the difference between merge and join.

# %% [markdown]
# Merge allows you to combine two datasets based on any specific column that the datasets have in common. Join allows you to combine them based on their indices.

# %% [markdown]
# ### Part 4: Divide dataset into two dataframes

# %%
61839 / 2

# %%
divided_df_1 = merged_dfs.head(30919)
divided_df_2 = merged_dfs.tail(30920)

# %% [markdown]
# ### Part 5: Concat the two dataframes

# %%
concat_df = pd.concat([divided_df_1, divided_df_2])
concat_df

# %% [markdown]
# ### Part 6: Handle duplicates

# %%
concat_df.duplicated().sum()
