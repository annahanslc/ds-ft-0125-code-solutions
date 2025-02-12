# %% [markdown]
# # telecom-db
#
# Retrieve table names from the database (telecom.db).
#
# Join all tables in the database on the primary key.
#
# Find unique occurrences of event_type and severity in the table from #2 using an SQL query.
#
# Find how many occurrences there are of each fault_severity in the table from #2 using an SQL query.

# %%
import pandas as pd
import sqlite3

# %% [markdown]
# # 1. Retrieve table names from the database (telecom.db).

# %%
filepath = ""

conn = sqlite3.connect(filepath + 'telecom.db')

# %%
q = """
  SELECT name
  FROM sqlite_master
  WHERE type = 'table'
"""

pd.read_sql(q, conn)

# %% [markdown]
# # 2. Join all tables in the database on the primary key.

# %%
q = """
  SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
  FROM train t
  LEFT OUTER JOIN event_type et
  ON t.id = et.id
  LEFT OUTER JOIN severity_type st
  ON t.id = st.id
  LEFT OUTER JOIN resource_type rt
  ON t.id = rt.id
  LEFT OUTER JOIN log_feature lf
  ON t.id = lf.id
"""

joined_table = pd.read_sql(q, conn)
joined_table

# %% [markdown]
# # 3. Find unique occurrences of event_type and severity in the table from #2 using an SQL query.

# %%
q = """
  WITH temp as(
  SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
  FROM train t
  LEFT OUTER JOIN event_type et
  ON t.id = et.id
  LEFT OUTER JOIN severity_type st
  ON t.id = st.id
  LEFT OUTER JOIN resource_type rt
  ON t.id = rt.id
  LEFT OUTER JOIN log_feature lf
  ON t.id = lf.id)

  SELECT fault_severity, event_type
  FROM temp
  GROUP BY fault_severity, event_type
"""

pd.read_sql(q, conn)

# %% [markdown]
# # 4. Find how many occurrences there are of each fault_severity in the table from #2 using an SQL query.

# %%
q = """
  WITH temp as(
  SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
  FROM train t
  LEFT OUTER JOIN event_type et
  ON t.id = et.id
  LEFT OUTER JOIN severity_type st
  ON t.id = st.id
  LEFT OUTER JOIN resource_type rt
  ON t.id = rt.id
  LEFT OUTER JOIN log_feature lf
  ON t.id = lf.id)

  SELECT fault_severity, COUNT(*)
  FROM temp
  GROUP BY fault_severity
"""

pd.read_sql(q, conn)

# %%
