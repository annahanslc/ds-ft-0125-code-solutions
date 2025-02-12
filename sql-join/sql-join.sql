-- sql-join

-- Write a SQL query to left outer join all the tables in the telecom database.
-- Note: Only ”id”, “location”, “fault_severity”, “event_type”, “severity_type”, 
-- “resource_type”, “log_feature”, “volume” columns will be included.

-- Step 1) check which table each column is located
SELECT COLUMN_NAME, TABLE_NAME
FROM information_schema.columns
WHERE COLUMN_NAME = "id";


-- Step 2) write the query
SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
FROM train t
LEFT OUTER JOIN event_type et
ON t.id = et.id
LEFT OUTER JOIN severity_type st 
ON et.id = st.id
LEFT OUTER JOIN resource_type rt
ON st.id = rt.id
LEFT OUTER JOIN log_feature lf
ON rt.id = lf.id;