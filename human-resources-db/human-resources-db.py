# %% [markdown]
# # human-resources-db

# %% [markdown]
# ### 1. Retrieve table names from the database (hr.db)

# %%
import pandas as pd
import sqlite3

# %%
filepath = ""

conn = sqlite3.connect(filepath + 'hr.db')

# %%
q = "SELECT name FROM sqlite_master WHERE type = 'table'"

pd.read_sql(q, conn)

# %%
q = """
  SELECT *
  FROM employee
  LIMIT 5
"""

pd.read_sql(q, conn)

# %% [markdown]
# ### 2. Retrieve EmployeeNumber, Department, Age, Gender, and Attrition for employees in sales department from the Employee table; save that information into a dataframe named ‘sales’.

# %%
q = """
  SELECT EmployeeNumber, Department, Age, Gender, Attrition
  FROM employee
  WHERE Department = "Sales"
"""

sales = pd.read_sql(q, conn)
sales

# %% [markdown]
# ### 3. Retrieve EmployeeNumber, EducationField, Age, Gender, and Attrition for employees in the Life Sciences field from the Employee table, save that information into a dataframe named ‘field’.

# %%
q = """
  SELECT EmployeeNumber, EducationField, Age, Gender, Attrition
  FROM employee
  WHERE EducationField = "Life Sciences"
"""

field = pd.read_sql(q, conn)
field

# %% [markdown]
# # 4. Save the two dataframes as tables in the database, and then join the tables on the primary key.

# %%
sales.to_sql('sales_table', conn, if_exists='replace', index=False)

# %%
q = """
  SELECT *
  FROM sales_table
"""

pd.read_sql(q, conn)

# %%
field.to_sql('field_table', conn, if_exists='replace', index=False)

# %%
q = """
  SELECT *
  FROM field_table
"""

pd.read_sql(q, conn)

# %%
q = """
  SELECT *
  FROM sales_table st
  INNER JOIN field_table ft
  ON st.EmployeeNumber = ft.EmployeeNumber
"""

pd.read_sql(q, conn)

# %%
