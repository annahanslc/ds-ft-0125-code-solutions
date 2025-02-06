# %% [markdown]
# # Days to Thanksgiving
#
# Exercise
# Calculate how many days there are until the next Thanksgiving.

# %%
import datetime as dt
import pytz

# %% [markdown]
# ##### Step 1: Get today's date, timezone aware

# %%
pst = pytz.timezone('US/Pacific')
aware_time_today = dt.datetime.now().astimezone(pst)
aware_time_today

# %% [markdown]
# ##### Step 2: Get the date of this year's Thanksgiving, timezone aware

# %%
thanksgiving_this_year = dt.datetime(2025,11,27).astimezone(pst)
thanksgiving_this_year

# %% [markdown]
# ##### Step 3: Calculate the difference between now and Thanksgivings

# %%
until_thanksgiving = thanksgiving_this_year - aware_time_today
until_thanksgiving

# %% [markdown]
# ##### Step 4: Get the number of days from the timedelta object

# %%
until_thanksgiving.days
