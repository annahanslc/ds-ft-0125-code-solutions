# %% [markdown]
# # Custom Datetimes
#
# Exercise
# Parse these custom datetime formats:
# 03/23/21
# 23/03/2021
# March 23rd, 2021 13:01 US/Pacific
# 1:01pm 23rd March, 2021 Europe/London
# 1616482800
# 2021-03-23T12:00:53.034-07:00

# %%
import datetime as dt

# %% [markdown]
# ##### Part 1: 03/23/21

# %%
time_1 = "03/23/21"
parsed_time_1 = dt.datetime.strptime(time_1,'%m/%d/%y')
print(parsed_time_1)

# %% [markdown]
# ##### Part 2: 23/03/2021

# %%
time_2 = "23/03/2021"
parsed_time_2 = dt.datetime.strptime(time_2, '%d/%m/%Y')
print(parsed_time_2)

# %% [markdown]
# ##### Part 3: March 23rd, 2021 13:01 US/Pacific
#
# Take of timezone and parse separately, using pytz

# %%
import pytz
# Step 1: Separate out the time zone
time_3 = "March 23rd, 2021 13:01 US/Pacific"
time_3_tz = time_3.split(' ')[-1]
print(time_3_tz)

# Step 2: Clean the string & remove time zone
cleaned_time_3 = time_3.replace("rd", "").replace("US/Pacific","")
print(cleaned_time_3)

# Step 3: Parse the string
parsed_time_3 = dt.datetime.strptime(cleaned_time_3, '%B %d, %Y %H:%M ')

# Step 4: Add the timezone back using pytz and print result
print(parsed_time_3.astimezone(pytz.timezone(time_3_tz)))

# Step 5: Double check the type to make sure it is datetime format
type(parsed_time_3)

# %% [markdown]
# ##### Part 4: 1:01pm 23rd March, 2021 Europe/London

# %%
time_4 = "1:01pm 23rd March, 2021 Europe/London"
cleaned_time_4 = time_4.replace('rd','').replace('Europe/London','GMT').replace('pm','PM')
print(cleaned_time_4)
parsed_time_4 = dt.datetime.strptime(cleaned_time_4, '%I:%M%p %d %B, %Y %Z')
print(parsed_time_4)

# %% [markdown]
# ##### Part 5: 1616482800

# %%
time_5 = "1616482800"
converted_unix = dt.datetime.fromtimestamp(int(time_5))
print(converted_unix)

# %% [markdown]
# ##### Part 6: 2021-03-23T12:00:53.034-07:00

# %%
time_6 = "2021-03-23T12:00:53.034-07:00"
cleaned_time_6 = time_6.replace('T',' ')
parsed_time_6 = dt.datetime.strptime(cleaned_time_6, '%Y-%m-%d %H:%M:%S.%f%z')
print(parsed_time_6)

# %%
