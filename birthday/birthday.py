# %% [markdown]
# # Birthday
#
# Exercise
# Write a function to calculate the age of a person based on the person's date of birth.

# %%
import datetime as dt
import pytz

# %% [markdown]
# ### Step 1: Get the current time

# %%
pst = pytz.timezone('US/Pacific')
aware_time_now = dt.datetime.now(tz=pst)
aware_time_now

# %% [markdown]
# ### Step 2: Define a function that calculates age based on user's input of their date of birth

# %%
def calculate_age():
  try:
    year, month, day = input("Enter your DOB as YYYY,MM,DD").split(',')     # obtain user input
    birthdate = [int(year), int(month), int(day)]                           # put user input into a list and as integers
    age_aware = dt.datetime(*birthdate).astimezone(pst)                     # add timezone
    age_in_days = aware_time_now - age_aware                                # find the difference in time
    age_in_years = age_in_days.days // 365                                  # change days to whole years
    return age_in_years                                                     # return age
  except:
    print('Something went wrong, please make sure you entered your DOB as YYYY,MM,DD with commas in between')

# %% [markdown]
# ### Step 3: Run the function & check solution

# %%
age = calculate_age()
print(age)

# %%
age_wrong = calculate_age()
print(age_wrong)

# %%
