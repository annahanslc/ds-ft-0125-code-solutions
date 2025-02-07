# %% [markdown]
# # stroke-analysis
#
# Exercise
#
# Use .describe() to compute a variety of statistics on the whole data set at once.
#
# Filter .describe() to only compute statistics on factors with floating point number values.
#
# Use .groupby() to create a data frame grouping by the "stroke" factor.
#
# Use the "stroke" grouping to get only group where "stroke" is 1.
#
# Use .describe() to compute statistics on factors with floating point values for the data where "stroke" is 1.
#
# Filter .describe() to only compute statistics on factors with integer values, removing as much percentile data as possible.
#
# Create a data frame grouping by both the "hypertension" and "heart_disease" factors.
#
# Get the group where both "hypertension" and "heart_disease" are 1.
#
# Count the number of "id"s per group.
#
# Aggregate both the mean and standard deviation of "stroke" per group.

# %%
import pandas as pd

# %%
healthcare_dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
healthcare_dataset.head()

# %% [markdown]
# ##### Part 1: Use .describe() to compute a variety of statistics on the whole data set at once.

# %%
healthcare_dataset.describe(include='all')

# %% [markdown]
# ##### Part 2: Filter .describe() to only compute statistics on factors with floating point number values.

# %%
healthcare_dataset.describe()

# %% [markdown]
# ##### Part 3: Use .groupby() to create a data frame grouping by the "stroke" factor.

# %%
healthcare_dataset.groupby('stroke').count()

# %% [markdown]
# ##### Part 4: Use the "stroke" grouping to get only group where "stroke" is 1.

# %%
healthcare_dataset[healthcare_dataset['stroke']==1].groupby('stroke').count()

# %% [markdown]
# ##### Part 5: Use .describe() to compute statistics on factors with floating point values for the data where "stroke" is 1.

# %%
healthcare_dataset[healthcare_dataset['stroke']==1].describe()

# %% [markdown]
# ##### Part 6: Filter .describe() to only compute statistics on factors with integer values, removing as much percentile data as possible.

# %%
healthcare_dataset.describe(include=[int])

# %% [markdown]
# ##### Part 7: Create a data frame grouping by both the "hypertension" and "heart_disease" factors.

# %%
healthcare_dataset.groupby(["hypertension","heart_disease"]).count()

# %% [markdown]
# ##### Part 8: Get the group where both "hypertension" and "heart_disease" are 1.

# %%
healthcare_dataset[(healthcare_dataset["hypertension"]==1) & (healthcare_dataset["heart_disease"]==1)]

# %% [markdown]
# ##### Part 9: Count the number of "id"s per group.

# %%
healthcare_dataset.groupby(["hypertension","heart_disease"])["id"].count()

# %% [markdown]
# ##### Step 10: Aggregate both the mean and standard deviation of "stroke" per group.

# %%
healthcare_dataset.groupby(["hypertension","heart_disease"])["stroke"].describe()

# %%
