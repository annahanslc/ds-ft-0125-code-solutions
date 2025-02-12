# %% [markdown]
# # stock-db
#
# Exercise
#
# Create two new tables in the database (stocks.sqlite), one with only 'MSFT' values for the Symbol feature, and one with only 'AAPL' values for the Symbol feature.
#
# Read the two new new tables in from the database using SQL to check if they were successfully created.
#
# For each new table in the database, query for rows containing the Maximum and Minimum dates, and save those rows as new pandas data frames (2 rows per dataframe).
#
# For each new table in the database, query for values greater than 50 in the Open feature, and save those as new pandas data frames.

# %%
import pandas as pd
import sqlite3

filepath = ""
conn = sqlite3.connect(filepath + 'stocks.sqlite')

# %%
q = """
  SELECT *
  FROM sqlite_master
  WHERE type='table'
"""

pd.read_sql(q, conn)

# %%
q = """
  SELECT *
  FROM STOCK_DATA
"""

pd.read_sql(q, conn)

# %% [markdown]
# ### 1. Create two new tables in the database (stocks.sqlite), one with only 'MSFT' values for the Symbol feature, and one with only 'AAPL' values for the Symbol feature.

# %% [markdown]
# # MSFT

# %%
q = """
  SELECT *
  FROM STOCK_DATA
  WHERE Symbol = 'MSFT'
"""
msft_df = pd.read_sql(q, conn)
msft_df

# %%
q = """
  PRAGMA table_info(STOCK_DATA);
"""

pd.read_sql(q, conn)

# %%
q = """
  CREATE TABLE msft_table
    (id INTEGER, Date text, Open real, High real, Low real, Close real, Volume integer, Adj Close real, Symbol text)
"""

conn.execute(q)
conn.commit()

# %%
q = """
  INSERT INTO msft_table
    (id, Date, Open, High, Low, Close, Volume, Adj, Symbol)
    SELECT *
    FROM STOCK_DATA
    WHERE Symbol = 'MSFT'
"""

conn.execute(q)
conn.commit()


# %% [markdown]
# ### Check if MSFT table was successfully created

# %%
q = """
  SELECT *
  FROM msft_table
"""

msft_table = pd.read_sql(q, conn)
msft_table

# %% [markdown]
# # AAPL

# %%
q = """
  SELECT *
  FROM STOCK_DATA
  WHERE Symbol = 'AAPL'
"""
aapl_df = pd.read_sql(q, conn)
aapl_df

# %%
q = """
  CREATE TABLE aapl_table
    (id INTEGER, Date text, Open real, High real, Low real, Close real, Volume integer, Adj real, Symbol text)
"""

conn.execute(q)
conn.commit()

# %%
q = """
  INSERT INTO aapl_table
    (id, Date, Open, High, Low, Close, Volume, Adj, Symbol)
    SELECT *
    FROM STOCK_DATA
    WHERE Symbol = 'AAPL'
"""

conn.execute(q)
conn.commit()

# %% [markdown]
# ### Check if AAPL table was successfully created

# %%
q = """
  SELECT *
  FROM aapl_table
"""

aapl_table = pd.read_sql(q, conn)
aapl_table

# %% [markdown]
# ### 3. For each new table in the database, query for rows containing the Maximum and Minimum dates, and save those rows as new pandas data frames (2 rows per dataframe).

# %%
q = """
  SELECT MAX(Date), MIN(Date)
  FROM msft_table
  ORDER BY Date DESC
"""

msft_max_min_dates = pd.read_sql(q, conn)
msft_max_min_dates

# %%
q = """
  SELECT *
  FROM msft_table
  WHERE Date = '2014-07-21' OR Date = '2000-01-03'
"""

msft_max_min_dates_df = pd.read_sql(q, conn).drop_duplicates()
msft_max_min_dates_df

# %%
q = """
  SELECT MAX(Date), MIN(Date)
  FROM aapl_table
  ORDER BY Date DESC
"""

aapl_max_min_dates_df = pd.read_sql(q, conn)
aapl_max_min_dates_df

# %%
q = """
  SELECT *
  FROM aapl_table
  WHERE Date = '2014-07-21' OR Date = '2000-01-03'
"""

aapl_max_min_dates_df = pd.read_sql(q, conn).drop_duplicates()
aapl_max_min_dates_df

# %% [markdown]
# # 4. For each new table in the database, query for values greater than 50 in the Open feature, and save those as new pandas data frames.

# %%
q = """
  SELECT *
  FROM msft_table
  WHERE Open > 50;
"""

msft_over50_open_df = pd.read_sql(q, conn)
msft_over50_open_df

# %%
type(msft_over50_open_df)

# %%
q = """
  SELECT *
  FROM aapl_table
  WHERE Open > 50;
"""

aapl_over50_open_df = pd.read_sql(q, conn)
aapl_over50_open_df

# %%
type(aapl_over50_open_df)

# %%
