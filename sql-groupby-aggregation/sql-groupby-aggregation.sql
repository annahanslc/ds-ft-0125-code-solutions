-- sql-groupby-aggregation
-- NOTE: Use the query from the sql-join assignment to create a temporary table to be used for the first two problems.

CREATE TEMPORARY TABLE dsstudent.temp_anna
	(id bigint, location varchar(20), fault_severity varchar(20), event_type varchar(20), severity_type varchar(20), resource_type varchar(20), log_feature varchar(20), volume bigint);

INSERT INTO dsstudent.temp_anna
	(id, location, fault_severity, event_type, severity_type, resource_type, log_feature, volume)
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

SELECT * FROM dsstudent.temp_anna;


-- 1. Write SQL statements to return data of the following questions:
-- -- For each location, what is the quantity of unique event types?

SELECT location, COUNT(DISTINCT(event_type))
FROM dsstudent.temp_anna
GROUP BY location;


-- -- What are the top 3 locations with the most volumes?

SELECT location, SUM(volume) as total_volume
FROM dsstudent.temp_anna
GROUP BY location
ORDER BY total_volume DESC;


-- 2. Write SQL statements to return data of the following questions:
-- -- For each fault severity, what is the quantity of unique locations?

SELECT fault_severity, COUNT(DISTINCT(location)) num_of_unique_locations
FROM dsstudent.temp_anna
GROUP BY fault_severity;


-- -- From the query result above, what is the quantity of unique locations with 
-- -- the fault_severity greater than 1?

SELECT fault_severity, COUNT(DISTINCT(location)) num_of_unique_locations
FROM dsstudent.temp_anna
GROUP BY fault_severity
HAVING fault_severity > 1;


-- 3. Write a SQL query to return the minimum, maximum, average of the field “Age” 
-- for each “Attrition” groups from the “hr” database.

SELECT Attrition, MIN(Age), MAX(Age), AVG(Age)
FROM employee
GROUP BY Attrition;


-- 4. Write a SQL query to return the “Attrition”, “Department” and the number 
-- of records from the ”hr” database for each group in the “Attrition” and “Department.” 
-- Sort the returned table by the “Attrition” and “Department” fields in ascending order.

SELECT Attrition, Department, COUNT(*) num_quantity
FROM employee
GROUP BY Attrition, Department
ORDER BY Attrition, Department;


-- 5. From Question #4, can you return the results where the “num_quantity” 
-- is greater than 100 records?

SELECT Attrition, Department, COUNT(*) num_quantity
FROM employee
GROUP BY Attrition, Department
HAVING num_quantity > 100
ORDER BY Attrition, Department;






